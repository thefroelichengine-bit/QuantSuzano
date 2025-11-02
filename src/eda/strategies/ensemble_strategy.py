"""Ensemble voting strategy with risk-reward execution."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from .voting import EnsembleVotingStrategy
from .risk_reward import RiskRewardExecutor
from ..automl.model_comparison import ModelComparator, compare_all_models
from ..utils_split import temporal_split
from ..backtest import simple_backtest
from ..metrics import sharpe_ratio, sortino_ratio, max_drawdown
from ..config import DATA_OUT


class EnsembleStrategy:
    """
    Complete ensemble strategy combining voting and risk-reward execution.
    
    Workflow:
    1. Train models on train data
    2. Generate predictions on TEST data only
    3. Vote on signals (TEST data only)
    4. Apply risk-reward filter (TEST data only)
    5. Execute trades and calculate metrics
    """
    
    def __init__(
        self,
        voting_method: str = 'majority',
        risk_reward_threshold: float = 1.5,
        z_threshold: float = 2.0,
        rolling_window: int = 60,
        min_vote_agreement: float = 0.6,
        max_correlation: float = 0.7,
    ):
        """
        Initialize ensemble strategy.
        
        Parameters
        ----------
        voting_method : str
            'majority', 'weighted', or 'threshold'
        risk_reward_threshold : float
            Minimum risk-reward ratio to execute trade
        z_threshold : float
            Z-score threshold for signal generation
        rolling_window : int
            Window for rolling z-score calculation
        min_vote_agreement : float
            Minimum agreement for threshold voting
        max_correlation : float
            Maximum correlation allowed for risk features
        """
        self.voting_method = voting_method
        self.risk_reward_threshold = risk_reward_threshold
        self.z_threshold = z_threshold
        self.rolling_window = rolling_window
        self.min_vote_agreement = min_vote_agreement
        self.max_correlation = max_correlation
        
        self.voting_strategy = None
        self.risk_executor = None
        self.comparator = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.full_df = None  # Full dataset for risk feature selection
        self.feature_cols = None
        self.target_col = None
        
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'suzb_r',
        feature_cols: List[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        """
        Train models and prepare strategy.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        target_col : str
            Target returns column
        feature_cols : list, optional
            Feature columns (auto-detected if None)
        train_ratio : float
            Training set proportion
        val_ratio : float
            Validation set proportion
        """
        print("\n" + "=" * 70)
        print("ENSEMBLE STRATEGY - TRAINING PHASE")
        print("=" * 70)
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [
                col for col in df.columns
                if col.endswith('_r') and col != target_col
                and not col.startswith(target_col.replace('_r', ''))
            ]
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Store full dataset for risk feature selection (uncorrelated features)
        self.full_df = df.copy()
        
        # Temporal split (critical for no data leakage)
        data = df[[target_col] + feature_cols].dropna()
        self.train_df, self.val_df, self.test_df = temporal_split(
            data,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        
        print(f"\n[SPLIT] Temporal split:")
        print(f"  Train: {len(self.train_df)} samples ({self.train_df.index.min()} to {self.train_df.index.max()})")
        print(f"  Val:   {len(self.val_df)} samples ({self.val_df.index.min()} to {self.val_df.index.max()})")
        print(f"  Test:  {len(self.test_df)} samples ({self.test_df.index.min()} to {self.test_df.index.max()})")
        
        # Prepare train/val for model comparison
        X_train = self.train_df[feature_cols].values
        y_train = self.train_df[target_col].values
        X_val = self.val_df[feature_cols].values
        y_val = self.val_df[target_col].values
        
        # Train models on TRAIN data only
        print(f"\n[MODEL TRAINING] Training models on train data...")
        self.comparator = ModelComparator()
        self.comparator.add_models()
        self.comparator.fit_all(X_train, y_train, X_val, y_val)
        
        # Initialize voting strategy
        self.voting_strategy = EnsembleVotingStrategy(
            voting_method=self.voting_method,
            z_threshold=self.z_threshold,
            rolling_window=self.rolling_window,
            min_vote_agreement=self.min_vote_agreement,
        )
        
        # Fit voting strategy with TEST data only
        self.voting_strategy.fit(
            self.comparator,
            self.test_df,
            feature_cols,
            target_col
        )
        
        # Initialize risk-reward executor
        self.risk_executor = RiskRewardExecutor(
            risk_reward_threshold=self.risk_reward_threshold,
            vol_window=self.rolling_window,
            max_correlation=self.max_correlation,
        )
        
        print("\n[FIT] Ensemble strategy ready")
        print(f"  Models trained: {len(self.voting_strategy.models)}")
        print(f"  Voting method: {self.voting_method}")
        print(f"  Risk-reward threshold: {self.risk_reward_threshold}")
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate signals using voting and risk-reward execution on TEST data only.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with voted signals, executed signals, and metrics
        """
        if self.voting_strategy is None:
            raise ValueError("Must call fit() first")
        
        print("\n" + "=" * 70)
        print("ENSEMBLE STRATEGY - SIGNAL GENERATION (TEST DATA ONLY)")
        print("=" * 70)
        
        # Generate voted signals from TEST data only
        signals_df, voted_signal = self.voting_strategy.generate_voted_signals()
        
        # Apply risk-reward filter on TEST data only
        # Use full dataset for risk feature selection (to find uncorrelated features)
        # But only TEST data for execution
        # First, select risk features using full dataset
        if len(self.risk_executor.risk_features) == 0:
            self.risk_executor.select_risk_features(
                self.full_df,
                self.target_col
            )
        
        # Execute strategy with risk-reward filter (using test data only)
        # Need full feature set for risk calculation, but restrict to test dates
        test_data_full = self.full_df.loc[self.test_df.index].copy()
        
        executed_df = self.risk_executor.execute_strategy(
            test_data_full,  # Pass test data with all features for risk calculation
            voted_signal,
            signals_df,
            target_col=self.target_col
        )
        
        # Combine all results
        result_df = signals_df.copy()
        for col in executed_df.columns:
            if col not in result_df.columns:
                result_df[col] = executed_df[col]
            else:
                # Rename to avoid conflicts
                result_df[f'exec_{col}'] = executed_df[col]
        
        return result_df
    
    def backtest(
        self,
        signals_df: pd.DataFrame = None,
        returns_col: str = None,
        transaction_cost: float = 0.001,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run backtest on executed signals.
        
        Parameters
        ----------
        signals_df : pd.DataFrame, optional
            Signals DataFrame (generated if None)
        returns_col : str, optional
            Returns column name
        transaction_cost : float
            Transaction cost per trade
        
        Returns
        -------
        metrics : dict
            Performance metrics
        results : pd.DataFrame
            Detailed backtest results
        """
        if signals_df is None:
            signals_df = self.generate_signals()
        
        if returns_col is None:
            returns_col = self.target_col
        
        # Prepare backtest data
        backtest_df = pd.DataFrame(index=signals_df.index)
        backtest_df['signal'] = signals_df['executed_signal'].fillna(0)
        backtest_df[returns_col] = signals_df['actual'].fillna(0)
        
        # Run backtest
        print(f"\n[BACKTEST] Running backtest with transaction cost={transaction_cost}...")
        metrics, results = simple_backtest(
            backtest_df,
            signal_col='signal',
            returns_col=returns_col,
            transaction_cost=transaction_cost,
        )
        
        # Add additional metrics
        returns = results['strategy_returns_net'].dropna()
        if len(returns) > 0:
            metrics['sharpe_ratio'] = sharpe_ratio(returns)
            metrics['sortino_ratio'] = sortino_ratio(returns)
            metrics['max_drawdown'] = max_drawdown(returns)
            
            # Calculate additional metrics
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) > 0:
                metrics['avg_win'] = positive_returns.mean()
            else:
                metrics['avg_win'] = 0.0
            
            if len(negative_returns) > 0:
                metrics['avg_loss'] = negative_returns.mean()
            else:
                metrics['avg_loss'] = 0.0
            
            if abs(metrics['avg_loss']) > 1e-6:
                metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            else:
                metrics['win_loss_ratio'] = 0.0
        
        print("\n[BACKTEST] Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return metrics, results
    
    def compare_with_single_models(
        self,
        signals_df: pd.DataFrame = None,
        transaction_cost: float = 0.001,
    ) -> pd.DataFrame:
        """
        Compare ensemble strategy performance vs. individual models.
        
        Parameters
        ----------
        signals_df : pd.DataFrame, optional
            Signals DataFrame
        transaction_cost : float
            Transaction cost per trade
        
        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each model
        """
        if signals_df is None:
            signals_df = self.generate_signals()
        
        print("\n" + "=" * 70)
        print("ENSEMBLE STRATEGY - MODEL COMPARISON")
        print("=" * 70)
        
        comparison_results = []
        
        # Ensemble strategy
        ensemble_metrics, _ = self.backtest(signals_df, transaction_cost=transaction_cost)
        comparison_results.append({
            'model': 'Ensemble (Voted + Risk-Reward)',
            **ensemble_metrics
        })
        
        # Individual model signals
        signal_cols = [col for col in signals_df.columns if col.endswith('_signal')]
        
        for signal_col in signal_cols:
            model_name = signal_col.replace('_signal', '')
            
            try:
                # Backtest individual model signal
                model_df = pd.DataFrame(index=signals_df.index)
                model_df['signal'] = signals_df[signal_col].fillna(0)
                model_df[self.target_col] = signals_df['actual'].fillna(0)
                
                model_metrics, _ = simple_backtest(
                    model_df,
                    signal_col='signal',
                    returns_col=self.target_col,
                    transaction_cost=transaction_cost,
                )
                
                comparison_results.append({
                    'model': model_name,
                    **model_metrics
                })
                
            except Exception as e:
                print(f"  WARNING: Failed to backtest {model_name}: {e}")
                continue
        
        comparison_df = pd.DataFrame(comparison_results)
        
        print("\n[COMPARISON] Model Performance:")
        if 'total_strategy_return' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('total_strategy_return', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_results(
        self,
        signals_df: pd.DataFrame,
        backtest_metrics: Dict,
        backtest_results: pd.DataFrame,
        comparison_df: pd.DataFrame = None,
        output_dir: Path = None,
    ):
        """
        Save all results to files.
        
        Parameters
        ----------
        signals_df : pd.DataFrame
            Signals DataFrame
        backtest_metrics : dict
            Backtest metrics
        backtest_results : pd.DataFrame
            Backtest results
        comparison_df : pd.DataFrame, optional
            Model comparison DataFrame
        output_dir : Path, optional
            Output directory
        """
        if output_dir is None:
            output_dir = DATA_OUT
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[SAVE] Saving results to {output_dir}...")
        
        # Save signals
        signals_path = output_dir / "ensemble_signals.parquet"
        signals_df.to_parquet(signals_path)
        print(f"  Signals: {signals_path}")
        
        # Save backtest results
        backtest_path = output_dir / "ensemble_backtest.parquet"
        backtest_results.to_parquet(backtest_path)
        print(f"  Backtest: {backtest_path}")
        
        # Save metrics
        metrics_path = output_dir / "ensemble_metrics.csv"
        pd.Series(backtest_metrics).to_csv(metrics_path)
        print(f"  Metrics: {metrics_path}")
        
        # Save comparison
        if comparison_df is not None:
            comparison_path = output_dir / "ensemble_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"  Comparison: {comparison_path}")
        
        # Generate plots
        try:
            from ..plots_validation import generate_ensemble_plots
            plots_dir = output_dir / "plots" / "strategies"
            generate_ensemble_plots(signals_df, comparison_df, plots_dir)
        except Exception as e:
            print(f"  WARNING: Failed to generate plots: {e}")

