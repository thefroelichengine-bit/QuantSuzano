"""Vote-based signal generation from multiple models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils_split import temporal_split
from ..config import ROLL_Z, Z_THRESHOLD


class EnsembleVotingStrategy:
    """
    Generate trading signals by voting across multiple models.
    
    Uses TEST data only to avoid data leakage. Converts model predictions
    to signals via z-score thresholds.
    """
    
    def __init__(
        self,
        voting_method: str = 'majority',
        z_threshold: float = 2.0,
        rolling_window: int = 60,
        min_vote_agreement: float = 0.6,
    ):
        """
        Initialize voting strategy.
        
        Parameters
        ----------
        voting_method : str
            'majority', 'weighted', or 'threshold'
        z_threshold : float
            Z-score threshold for signal generation
        rolling_window : int
            Window for rolling z-score calculation
        min_vote_agreement : float
            Minimum agreement required for threshold voting (0-1)
        """
        self.voting_method = voting_method
        self.z_threshold = z_threshold
        self.rolling_window = rolling_window
        self.min_vote_agreement = min_vote_agreement
        self.models = {}
        self.model_weights = {}
        self.test_df = None
        self.feature_cols = None
        self.target_col = None
        
    def fit(
        self,
        comparator,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'suzb_r',
    ):
        """
        Store models and test data from ModelComparator.
        
        Parameters
        ----------
        comparator : ModelComparator
            Fitted ModelComparator with trained models
        test_df : pd.DataFrame
            TEST data (must be separate from train/val)
        feature_cols : list
            Feature column names
        target_col : str
            Target column name
        """
        self.test_df = test_df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Store models that are valid
        for name, model in comparator.models.items():
            if comparator.results.get(name) is not None:
                self.models[name] = model
                
                # Store weights based on validation performance
                if self.voting_method == 'weighted':
                    val_r2 = comparator.results[name]['val']['R2']
                    # Convert to positive weight (handle negative R2)
                    self.model_weights[name] = max(0, val_r2) + 0.01
                else:
                    self.model_weights[name] = 1.0
        
        # Normalize weights
        if self.voting_method == 'weighted':
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                self.model_weights = {
                    k: v / total_weight for k, v in self.model_weights.items()
                }
        
        print(f"\n[VOTING] Initialized with {len(self.models)} models")
        print(f"  Voting method: {self.voting_method}")
        if self.voting_method == 'weighted':
            print(f"  Model weights:")
            for name, weight in sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {name}: {weight:.4f}")
    
    def generate_model_signals(self) -> pd.DataFrame:
        """
        Generate individual signals from each model's predictions on TEST data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: model_name_signal, model_name_zscore, etc.
        """
        if self.test_df is None:
            raise ValueError("Must call fit() first with test data")
        
        # Prepare test features and target
        X_test = self.test_df[self.feature_cols].values
        y_test = self.test_df[self.target_col].values
        
        signals_df = pd.DataFrame(index=self.test_df.index)
        signals_df['actual'] = y_test
        
        print(f"\n[VOTING] Generating signals from {len(self.models)} models on TEST data...")
        
        for model_name, model in self.models.items():
            try:
                # Get predictions on TEST data only
                predictions = model.predict(X_test)
                predictions_series = pd.Series(predictions, index=self.test_df.index)
                
                # Calculate spread (actual - predicted)
                spread = y_test - predictions
                spread_series = pd.Series(spread, index=self.test_df.index)
                
                # Calculate rolling z-score
                rolling_mean = spread_series.rolling(
                    window=self.rolling_window,
                    min_periods=self.rolling_window // 2
                ).mean()
                rolling_std = spread_series.rolling(
                    window=self.rolling_window,
                    min_periods=self.rolling_window // 2
                ).std()
                
                zscore = (spread_series - rolling_mean) / rolling_std
                zscore = zscore.replace([np.inf, -np.inf], np.nan)
                
                # Convert z-score to signal
                signal = pd.Series(0, index=zscore.index)
                signal[zscore > self.z_threshold] = -1  # Short (overvalued)
                signal[zscore < -self.z_threshold] = 1  # Long (undervalued)
                
                # Store results
                signals_df[f'{model_name}_prediction'] = predictions_series
                signals_df[f'{model_name}_zscore'] = zscore
                signals_df[f'{model_name}_signal'] = signal
                
                # Count signals
                long_count = (signal == 1).sum()
                short_count = (signal == -1).sum()
                neutral_count = (signal == 0).sum()
                
                print(f"  {model_name}: Long={long_count}, Short={short_count}, Neutral={neutral_count}")
                
            except Exception as e:
                print(f"  WARNING: Failed to generate signals for {model_name}: {e}")
                continue
        
        return signals_df
    
    def vote_signals(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Aggregate individual model signals using voting method.
        
        Parameters
        ----------
        signals_df : pd.DataFrame
            DataFrame with individual model signals
        
        Returns
        -------
        pd.Series
            Voted signal series (-1, 0, 1)
        """
        signal_cols = [col for col in signals_df.columns if col.endswith('_signal')]
        
        if len(signal_cols) == 0:
            raise ValueError("No signal columns found in DataFrame")
        
        # Extract signal matrix
        signal_matrix = signals_df[signal_cols].values
        
        # Handle NaN values (set to 0)
        signal_matrix = np.nan_to_num(signal_matrix, nan=0.0)
        
        print(f"\n[VOTING] Aggregating signals from {len(signal_cols)} models...")
        
        if self.voting_method == 'majority':
            # Simple majority vote
            voted_signal = np.zeros(len(signal_matrix))
            for i in range(len(signal_matrix)):
                votes = signal_matrix[i, :]
                # Count votes
                long_votes = np.sum(votes == 1)
                short_votes = np.sum(votes == -1)
                
                if long_votes > short_votes and long_votes > 0:
                    voted_signal[i] = 1
                elif short_votes > long_votes and short_votes > 0:
                    voted_signal[i] = -1
                else:
                    voted_signal[i] = 0
        
        elif self.voting_method == 'weighted':
            # Weighted voting
            voted_signal = np.zeros(len(signal_matrix))
            for i in range(len(signal_matrix)):
                votes = signal_matrix[i, :]
                weighted_long = 0.0
                weighted_short = 0.0
                
                for j, model_name in enumerate([col.replace('_signal', '') for col in signal_cols]):
                    if model_name in self.model_weights:
                        weight = self.model_weights[model_name]
                        if votes[j] == 1:
                            weighted_long += weight
                        elif votes[j] == -1:
                            weighted_short += weight
                
                if weighted_long > weighted_short and weighted_long > 0:
                    voted_signal[i] = 1
                elif weighted_short > weighted_long and weighted_short > 0:
                    voted_signal[i] = -1
                else:
                    voted_signal[i] = 0
        
        elif self.voting_method == 'threshold':
            # Require minimum agreement
            voted_signal = np.zeros(len(signal_matrix))
            n_models = len(signal_cols)
            min_agreement_count = int(n_models * self.min_vote_agreement)
            
            for i in range(len(signal_matrix)):
                votes = signal_matrix[i, :]
                long_votes = np.sum(votes == 1)
                short_votes = np.sum(votes == -1)
                
                if long_votes >= min_agreement_count:
                    voted_signal[i] = 1
                elif short_votes >= min_agreement_count:
                    voted_signal[i] = -1
                else:
                    voted_signal[i] = 0
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        voted_series = pd.Series(voted_signal, index=signals_df.index, name='voted_signal')
        
        # Count final signals
        long_count = (voted_series == 1).sum()
        short_count = (voted_series == -1).sum()
        neutral_count = (voted_series == 0).sum()
        
        print(f"  Voted signals: Long={long_count}, Short={short_count}, Neutral={neutral_count}")
        
        return voted_series
    
    def generate_voted_signals(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full pipeline: generate individual signals and vote.
        
        Returns
        -------
        signals_df : pd.DataFrame
            DataFrame with all model signals and predictions
        voted_signal : pd.Series
            Final voted signal
        """
        signals_df = self.generate_model_signals()
        voted_signal = self.vote_signals(signals_df)
        
        signals_df['voted_signal'] = voted_signal
        
        return signals_df, voted_signal

