"""Pipeline orchestrator for coordinating data collection."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from ..config import ROOT, DATA_RAW
from ..scrapers import registry
from .validator import DataValidator
from .versioning import DataVersionManager
from .incremental import IncrementalUpdater


class DataPipeline:
    """
    Main pipeline orchestrator for automated data collection.
    
    Coordinates:
    - Multiple data sources
    - Scraping with retry logic
    - Data validation
    - Version management
    - Incremental updates
    """
    
    # Data source configuration
    SOURCES = {
        "ptax": {
            "scraper": "bcb_extended",
            "method": "fetch_ptax",
            "update_frequency_hours": 24,
            "required": True,
        },
        "selic": {
            "scraper": "bcb_extended",
            "method": "fetch_selic",
            "update_frequency_hours": 24,
            "required": True,
        },
        "suzb3": {
            "scraper": "yfinance",
            "method": "fetch_equity",
            "params": {"ticker": "SUZB3.SA", "column": "close"},
            "update_frequency_hours": 1,
            "required": True,
        },
        "climate_nasa": {
            "scraper": "nasa_power",
            "method": "fetch_aggregated",
            "params": {"locations": ["tres_lagoas", "imperatriz"]},
            "update_frequency_hours": 168,  # Weekly
            "required": False,
        },
        "climate_inmet": {
            "scraper": "inmet",
            "method": "fetch_aggregated",
            "params": {"stations": ["tres_lagoas_ms", "imperatriz_ma"]},
            "update_frequency_hours": 24,
            "required": False,
        },
    }
    
    def __init__(
        self,
        data_dir: Path = None,
        version_dir: Path = None,
        start_date: str = "2020-01-01"
    ):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        data_dir : Path, optional
            Directory for output data
        version_dir : Path, optional
            Directory for version storage
        start_date : str
            Default start date for historical data
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_RAW
        self.version_dir = Path(version_dir) if version_dir else ROOT / "data" / "versions"
        self.start_date = start_date
        
        # Initialize components
        self.validator = DataValidator()
        self.version_manager = DataVersionManager(self.version_dir)
        self.updater = IncrementalUpdater(self.version_manager)
        
        # Execution stats
        self.stats = {
            "start_time": None,
            "end_time": None,
            "sources_attempted": 0,
            "sources_succeeded": 0,
            "sources_failed": 0,
            "errors": [],
        }
        
        print(f"[PIPELINE] Initialized")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Version dir: {self.version_dir}")
        print(f"  Start date: {self.start_date}")
    
    def fetch_source(
        self,
        source_name: str,
        force_full: bool = False,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single source.
        
        Parameters
        ----------
        source_name : str
            Source name from SOURCES config
        force_full : bool
            Force full historical fetch (ignore incremental)
        use_cache : bool
            Use scraper cache
        
        Returns
        -------
        pd.DataFrame or None
            Fetched data
        """
        if source_name not in self.SOURCES:
            raise ValueError(f"Unknown source: {source_name}")
        
        config = self.SOURCES[source_name]
        
        print(f"\n{'='*60}")
        print(f"[PIPELINE] Fetching: {source_name}")
        print(f"{'='*60}")
        
        try:
            # Get scraper
            scraper = registry.get(config["scraper"])
            
            # Get fetch method
            method = getattr(scraper, config["method"])
            
            # Prepare parameters
            params = config.get("params", {})
            params["use_cache"] = use_cache
            
            # Determine if incremental update is needed
            if not force_full and self.updater.needs_update(
                source_name,
                max_age_hours=config["update_frequency_hours"]
            ):
                # Incremental update
                df = self.updater.update(
                    fetch_func=method,
                    source=source_name,
                    default_start=self.start_date,
                    **params
                )
            else:
                if force_full:
                    print(f"[PIPELINE] Force full fetch")
                else:
                    print(f"[PIPELINE] Data is fresh, loading from version")
                
                # Check if we have existing data
                df = self.version_manager.load_latest(source_name)
                
                if df is None or force_full:
                    # Full fetch
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    df = method(
                        start_date=self.start_date,
                        end_date=end_date,
                        **params
                    )
                    
                    # Save version
                    self.version_manager.save_version(
                        df,
                        source=source_name,
                        metadata={"update_type": "full"}
                    )
            
            # Validate
            validation = self.validator.validate(df, name=source_name)
            
            if not validation.passed:
                print(f"[PIPELINE] Validation failed for {source_name}")
                if config["required"]:
                    raise ValueError("Validation failed for required source")
            
            # Save to CSV (for backward compatibility)
            output_path = self.data_dir / f"{source_name}.csv"
            df.to_csv(output_path)
            print(f"[PIPELINE] Saved to {output_path}")
            
            return df
        
        except Exception as e:
            error_msg = f"{source_name}: {str(e)}"
            print(f"[PIPELINE] ERROR: {error_msg}")
            self.stats["errors"].append(error_msg)
            
            if config["required"]:
                raise
            
            return None
    
    def run(
        self,
        sources: List[str] = None,
        force_full: bool = False,
        use_cache: bool = True,
        continue_on_error: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run pipeline for multiple sources.
        
        Parameters
        ----------
        sources : list, optional
            List of source names (default: all)
        force_full : bool
            Force full historical fetch
        use_cache : bool
            Use scraper cache
        continue_on_error : bool
            Continue if a source fails
        
        Returns
        -------
        dict
            Dictionary of source_name -> DataFrame
        """
        self.stats["start_time"] = datetime.now()
        
        if sources is None:
            sources = list(self.SOURCES.keys())
        
        print(f"\n{'#'*60}")
        print(f"[PIPELINE] Starting data pipeline")
        print(f"[PIPELINE] Sources: {sources}")
        print(f"{'#'*60}\n")
        
        results = {}
        
        for source in sources:
            self.stats["sources_attempted"] += 1
            
            try:
                df = self.fetch_source(
                    source,
                    force_full=force_full,
                    use_cache=use_cache
                )
                
                if df is not None:
                    results[source] = df
                    self.stats["sources_succeeded"] += 1
                else:
                    self.stats["sources_failed"] += 1
            
            except Exception as e:
                self.stats["sources_failed"] += 1
                print(f"[PIPELINE] Failed to fetch {source}: {e}")
                
                if not continue_on_error:
                    raise
        
        self.stats["end_time"] = datetime.now()
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _print_summary(self):
        """Print execution summary."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        print(f"\n{'#'*60}")
        print(f"[PIPELINE] Execution Summary")
        print(f"{'#'*60}")
        print(f"Duration: {duration:.1f}s")
        print(f"Sources attempted: {self.stats['sources_attempted']}")
        print(f"Sources succeeded: {self.stats['sources_succeeded']}")
        print(f"Sources failed: {self.stats['sources_failed']}")
        
        if self.stats["errors"]:
            print(f"\nErrors:")
            for error in self.stats["errors"]:
                print(f"  - {error}")
        
        print(f"{'#'*60}\n")
    
    def get_merged_data(self) -> pd.DataFrame:
        """
        Load and merge all available data sources.
        
        Returns
        -------
        pd.DataFrame
            Merged data from all sources
        """
        dfs = []
        
        for source in self.SOURCES.keys():
            df = self.version_manager.load_latest(source)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No data available")
        
        # Merge all sources
        merged = pd.concat(dfs, axis=1)
        merged = merged.sort_index()
        
        print(f"[PIPELINE] Merged data: {len(merged)} rows, {len(merged.columns)} columns")
        
        return merged
    
    def cleanup_old_versions(self, keep_last_n: int = 10):
        """
        Clean up old versions for all sources.
        
        Parameters
        ----------
        keep_last_n : int
            Number of versions to keep per source
        """
        print(f"\n[PIPELINE] Cleaning up old versions (keep last {keep_last_n})")
        
        for source in self.SOURCES.keys():
            self.version_manager.cleanup_old_versions(source, keep_last_n)
        
        print("[PIPELINE] Cleanup complete")

