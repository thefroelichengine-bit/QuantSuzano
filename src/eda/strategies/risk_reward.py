"""Risk-reward decision model for trade execution."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RiskRewardExecutor:
    """
    Evaluate risk-reward ratios and decide whether to execute trades.
    
    Uses uncorrelated features to avoid data leakage. Only uses market-wide
    indicators, not SUZB3-specific returns or directly correlated features.
    """
    
    def __init__(
        self,
        risk_reward_threshold: float = 1.5,
        vol_window: int = 60,
        max_correlation: float = 0.7,
    ):
        """
        Initialize risk-reward executor.
        
        Parameters
        ----------
        risk_reward_threshold : float
            Minimum risk-reward ratio to execute trade (default 1.5)
        vol_window : int
            Window for rolling volatility calculation
        max_correlation : float
            Maximum allowed correlation with target returns for risk features
        """
        self.risk_reward_threshold = risk_reward_threshold
        self.vol_window = vol_window
        self.max_correlation = max_correlation
        self.risk_features = []
        
    def select_risk_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'suzb_r',
        exclude_patterns: List[str] = None,
    ) -> List[str]:
        """
        Select features for risk-reward calculation.
        
        Excludes:
        - Directly correlated features (correlation > max_correlation)
        - SUZB3-specific returns
        - Synthetic index and z-scores from SUZB3 models
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        target_col : str
            Target returns column
        exclude_patterns : list, optional
            Additional patterns to exclude (e.g., ['suzb', 'synthetic', 'zscore'])
        
        Returns
        -------
        list
            Selected risk feature column names
        """
        if exclude_patterns is None:
            exclude_patterns = ['suzb', 'synthetic', 'zscore', target_col.replace('_r', '')]
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Exclude patterns
        candidate_features = []
        for col in numeric_cols:
            exclude = False
            for pattern in exclude_patterns:
                if pattern.lower() in col.lower():
                    exclude = True
                    break
            if not exclude:
                candidate_features.append(col)
        
        # Calculate correlation with target returns
        target_returns = df[target_col].dropna()
        selected_features = []
        
        print(f"\n[RISK-REWARD] Selecting risk features (max correlation: {self.max_correlation})...")
        
        for col in candidate_features:
            try:
                feature_series = df[col].dropna()
                
                # Align indices
                common_idx = target_returns.index.intersection(feature_series.index)
                if len(common_idx) < 50:  # Need minimum observations
                    continue
                
                target_aligned = target_returns.loc[common_idx]
                feature_aligned = feature_series.loc[common_idx]
                
                # Calculate correlation
                corr = target_aligned.corr(feature_aligned)
                
                if abs(corr) <= self.max_correlation:
                    selected_features.append(col)
                    print(f"  ✓ {col} (corr={corr:.3f})")
                else:
                    print(f"  ✗ {col} (corr={corr:.3f}, too correlated)")
                    
            except Exception as e:
                continue
        
        if len(selected_features) == 0:
            print(f"  WARNING: No uncorrelated features found. Using default market indicators.")
            # Fallback to common market indicators
            fallback = ['ibov_r', 'imat_r', 'iagro_r', 'selic_r', 'ptax_r']
            selected_features = [col for col in fallback if col in df.columns]
        
        self.risk_features = selected_features
        print(f"  Selected {len(selected_features)} risk features")
        
        return selected_features
    
    def calculate_risk_reward_ratio(
        self,
        df: pd.DataFrame,
        predicted_returns: pd.Series,
        actual_returns: pd.Series,
    ) -> pd.Series:
        """
        Calculate risk-reward ratio for each time point.
        
        Risk-reward = |Expected Return| / Expected Risk
        
        Where:
        - Expected Return: |Predicted Return| (magnitude of expected move)
        - Expected Risk: Rolling volatility from uncorrelated market features
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with risk features
        predicted_returns : pd.Series
            Model predictions (expected returns)
        actual_returns : pd.Series
            Actual returns (for validation)
        
        Returns
        -------
        pd.Series
            Risk-reward ratios
        """
        # Calculate expected return (magnitude of predicted move)
        expected_return = np.abs(predicted_returns)
        
        # Calculate expected risk from market volatility
        # Use average volatility across selected risk features
        risk_values = []
        for feature in self.risk_features:
            if feature in df.columns:
                feature_returns = df[feature].dropna()
                # Calculate rolling volatility
                vol = feature_returns.rolling(window=self.vol_window, min_periods=self.vol_window // 2).std()
                # Annualize
                vol_annualized = vol * np.sqrt(252)
                risk_values.append(vol_annualized)
        
        if len(risk_values) == 0:
            # Fallback: use simple rolling volatility of target
            target_vol = actual_returns.rolling(
                window=self.vol_window,
                min_periods=self.vol_window // 2
            ).std() * np.sqrt(252)
            expected_risk = target_vol
        else:
            # Average across risk features
            risk_df = pd.concat(risk_values, axis=1)
            expected_risk = risk_df.mean(axis=1)
        
        # Calculate risk-reward ratio
        # Avoid division by zero
        risk_reward_ratio = expected_return / (expected_risk + 1e-6)
        
        # Store in DataFrame format
        result = pd.DataFrame(index=predicted_returns.index)
        result['expected_return'] = expected_return
        result['expected_risk'] = expected_risk
        result['risk_reward_ratio'] = risk_reward_ratio
        
        return result
    
    def evaluate_trade(
        self,
        voted_signal: pd.Series,
        risk_reward_df: pd.DataFrame,
        current_position: float = 0.0,
    ) -> pd.Series:
        """
        Decide whether to execute trade based on risk-reward ratio.
        
        Decision logic:
        - If risk-reward ratio > threshold: Execute signal
        - If risk-reward ratio <= threshold: Stay in current position (or exit)
        
        Parameters
        ----------
        voted_signal : pd.Series
            Voted signal (-1, 0, 1)
        risk_reward_df : pd.DataFrame
            DataFrame with risk_reward_ratio column
        current_position : float
            Current position state (-1, 0, 1)
        
        Returns
        -------
        pd.Series
            Executed signal after risk-reward filter (-1, 0, 1)
        """
        # Align indices
        common_idx = voted_signal.index.intersection(risk_reward_df.index)
        
        executed_signal = pd.Series(0.0, index=common_idx)
        risk_reward_aligned = risk_reward_df.loc[common_idx, 'risk_reward_ratio']
        signal_aligned = voted_signal.loc[common_idx]
        
        for i, idx in enumerate(common_idx):
            signal = signal_aligned.loc[idx]
            rr_ratio = risk_reward_aligned.loc[idx]
            
            if pd.isna(rr_ratio) or pd.isna(signal):
                # Default to staying in current position
                executed_signal.loc[idx] = current_position
                continue
            
            # Decision: Execute if risk-reward is favorable
            if rr_ratio >= self.risk_reward_threshold:
                # Execute the signal
                executed_signal.loc[idx] = signal
            else:
                # Risk-reward not favorable: stay in current position or exit
                if abs(current_position) > 0:
                    # Exit current position if risk-reward not favorable
                    executed_signal.loc[idx] = 0.0
                else:
                    # Stay flat
                    executed_signal.loc[idx] = 0.0
        
        return executed_signal
    
    def execute_strategy(
        self,
        df: pd.DataFrame,
        voted_signal: pd.Series,
        model_predictions: pd.DataFrame,
        target_col: str = 'suzb_r',
    ) -> pd.DataFrame:
        """
        Full pipeline: calculate risk-reward and execute trades.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with all features
        voted_signal : pd.Series
            Voted signal from ensemble
        model_predictions : pd.DataFrame
            DataFrame with model predictions (one column per model)
        target_col : str
            Target returns column
        
        Returns
        -------
        pd.DataFrame
            DataFrame with risk-reward metrics and executed signals
        """
        print(f"\n[RISK-REWARD] Executing strategy with threshold={self.risk_reward_threshold}...")
        
        # Select risk features if not already selected
        if len(self.risk_features) == 0:
            self.select_risk_features(df, target_col)
        
        # Calculate average prediction across models (as proxy for expected return)
        pred_cols = [col for col in model_predictions.columns if col.endswith('_prediction')]
        if len(pred_cols) > 0:
            # Use average of model predictions as expected return
            avg_prediction = model_predictions[pred_cols].mean(axis=1)
        else:
            # Fallback: use mean of 0 (no expected return)
            avg_prediction = pd.Series(0.0, index=voted_signal.index)
        
        # Get actual returns for calculation
        actual_returns = df[target_col].dropna()
        
        # Calculate risk-reward ratio
        risk_reward_df = self.calculate_risk_reward_ratio(
            df,
            avg_prediction,
            actual_returns,
        )
        
        # Track current position (for position management)
        current_position = 0.0
        executed_signals = []
        
        # Execute trades with position tracking
        for idx in voted_signal.index:
            if idx not in risk_reward_df.index:
                executed_signals.append(current_position)
                continue
            
            signal = voted_signal.loc[idx]
            rr_ratio = risk_reward_df.loc[idx, 'risk_reward_ratio']
            
            if pd.isna(rr_ratio) or pd.isna(signal):
                executed_signals.append(current_position)
                continue
            
            # Decision logic
            if rr_ratio >= self.risk_reward_threshold:
                # Execute signal
                executed_signals.append(signal)
                current_position = signal
            else:
                # Risk-reward not favorable: exit if holding, stay flat otherwise
                if abs(current_position) > 0:
                    executed_signals.append(0.0)
                    current_position = 0.0
                else:
                    executed_signals.append(0.0)
        
        executed_signal = pd.Series(executed_signals, index=voted_signal.index, name='executed_signal')
        
        # Combine results
        result_df = pd.DataFrame(index=voted_signal.index)
        result_df['voted_signal'] = voted_signal
        result_df['executed_signal'] = executed_signal
        
        # Add risk-reward metrics
        for col in ['expected_return', 'expected_risk', 'risk_reward_ratio']:
            if col in risk_reward_df.columns:
                result_df[col] = risk_reward_df[col]
        
        # Count executed vs. voted signals
        voted_long = (voted_signal == 1).sum()
        voted_short = (voted_signal == -1).sum()
        executed_long = (executed_signal == 1).sum()
        executed_short = (executed_signal == -1).sum()
        
        print(f"  Voted signals: Long={voted_long}, Short={voted_short}")
        print(f"  Executed signals: Long={executed_long}, Short={executed_short}")
        print(f"  Filtered out: {(voted_long + voted_short) - (executed_long + executed_short)} trades")
        
        return result_df

