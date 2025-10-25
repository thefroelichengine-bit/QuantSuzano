"""Incremental data updater for efficient data refresh."""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from .versioning import DataVersionManager


class IncrementalUpdater:
    """
    Manage incremental data updates to avoid re-fetching entire history.
    
    Strategy:
    - Track last successful update timestamp
    - Only fetch new data since last update
    - Merge with existing data
    - Handle gaps and overlaps
    """
    
    def __init__(self, version_manager: DataVersionManager):
        """
        Initialize incremental updater.
        
        Parameters
        ----------
        version_manager : DataVersionManager
            Version manager for tracking updates
        """
        self.version_manager = version_manager
    
    def get_update_range(
        self,
        source: str,
        default_start: str = "2020-01-01",
        overlap_days: int = 5
    ) -> tuple:
        """
        Determine date range for incremental update.
        
        Parameters
        ----------
        source : str
            Data source name
        default_start : str
            Default start date if no history exists
        overlap_days : int
            Number of days to overlap with existing data (for safety)
        
        Returns
        -------
        tuple
            (start_date, end_date) as strings
        """
        # Get latest version
        latest = self.version_manager.get_latest_version(source)
        
        if latest is None or not latest.end_date:
            # No history, fetch from default start
            start_date = default_start
            print(f"[INCREMENTAL] No history for {source}, fetching from {start_date}")
        else:
            # Fetch from last end date minus overlap
            last_end = pd.to_datetime(latest.end_date)
            start_date = (last_end - timedelta(days=overlap_days)).strftime("%Y-%m-%d")
            print(f"[INCREMENTAL] Updating {source} from {start_date} (last: {latest.end_date})")
        
        # End date is today
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        return start_date, end_date
    
    def merge_with_existing(
        self,
        new_data: pd.DataFrame,
        source: str,
        on_conflict: str = "new"
    ) -> pd.DataFrame:
        """
        Merge new data with existing historical data.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            Newly fetched data
        source : str
            Data source name
        on_conflict : str
            Strategy for overlapping dates: 'new', 'old', 'average'
        
        Returns
        -------
        pd.DataFrame
            Merged data
        """
        # Load existing data
        existing = self.version_manager.load_latest(source)
        
        if existing is None or len(existing) == 0:
            print(f"[INCREMENTAL] No existing data for {source}, using new data only")
            return new_data
        
        print(f"[INCREMENTAL] Merging with existing data ({len(existing)} rows)")
        
        # Concatenate
        combined = pd.concat([existing, new_data])
        
        # Handle duplicates based on index
        if on_conflict == "new":
            # Keep last (new data overrides old)
            merged = combined[~combined.index.duplicated(keep="last")]
        elif on_conflict == "old":
            # Keep first (old data preserved)
            merged = combined[~combined.index.duplicated(keep="first")]
        elif on_conflict == "average":
            # Average overlapping values
            merged = combined.groupby(combined.index).mean()
        else:
            raise ValueError(f"Unknown conflict strategy: {on_conflict}")
        
        # Sort by index
        merged = merged.sort_index()
        
        print(f"[INCREMENTAL] Merged data: {len(merged)} rows "
              f"({len(existing)} existing + {len(new_data)} new = {len(merged)} after dedup)")
        
        return merged
    
    def update(
        self,
        fetch_func: callable,
        source: str,
        default_start: str = "2020-01-01",
        overlap_days: int = 5,
        on_conflict: str = "new",
        **fetch_kwargs
    ) -> pd.DataFrame:
        """
        Perform incremental update.
        
        Parameters
        ----------
        fetch_func : callable
            Function to fetch data, signature: (start_date, end_date, **kwargs) -> DataFrame
        source : str
            Data source name
        default_start : str
            Default start date
        overlap_days : int
            Overlap days for safety
        on_conflict : str
            Conflict resolution strategy
        **fetch_kwargs
            Additional arguments for fetch function
        
        Returns
        -------
        pd.DataFrame
            Updated data
        """
        print(f"\n[INCREMENTAL] Starting update for {source}")
        
        # Determine update range
        start_date, end_date = self.get_update_range(
            source,
            default_start=default_start,
            overlap_days=overlap_days
        )
        
        # Fetch new data
        try:
            new_data = fetch_func(
                start_date=start_date,
                end_date=end_date,
                **fetch_kwargs
            )
        except Exception as e:
            print(f"[INCREMENTAL] Fetch failed for {source}: {e}")
            # Return existing data if fetch fails
            existing = self.version_manager.load_latest(source)
            if existing is not None:
                print(f"[INCREMENTAL] Using existing data")
                return existing
            raise
        
        # Merge with existing
        merged = self.merge_with_existing(new_data, source, on_conflict=on_conflict)
        
        # Save new version
        self.version_manager.save_version(
            merged,
            source=source,
            metadata={
                "update_type": "incremental",
                "new_rows": len(new_data),
                "total_rows": len(merged),
            }
        )
        
        print(f"[INCREMENTAL] Update complete for {source}")
        
        return merged
    
    def needs_update(
        self,
        source: str,
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if data needs updating based on age.
        
        Parameters
        ----------
        source : str
            Data source name
        max_age_hours : int
            Maximum age in hours before update needed
        
        Returns
        -------
        bool
            True if update is needed
        """
        latest = self.version_manager.get_latest_version(source)
        
        if latest is None:
            return True
        
        # Check age
        last_update = datetime.fromisoformat(latest.timestamp)
        age = datetime.now() - last_update
        
        needs_update = age.total_seconds() / 3600 > max_age_hours
        
        if needs_update:
            print(f"[INCREMENTAL] {source} needs update (age: {age.total_seconds()/3600:.1f}h)")
        else:
            print(f"[INCREMENTAL] {source} is fresh (age: {age.total_seconds()/3600:.1f}h)")
        
        return needs_update

