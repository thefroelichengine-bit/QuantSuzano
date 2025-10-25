"""Data versioning system for tracking changes."""

import json
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


@dataclass
class DataVersion:
    """Metadata for a data version."""
    version_id: str
    timestamp: str
    source: str
    rows: int
    columns: int
    hash: str
    start_date: str
    end_date: str
    metadata: Dict = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        return cls(**d)


class DataVersionManager:
    """
    Manage data versions with metadata tracking.
    
    Features:
    - Version history
    - Change detection
    - Rollback capability
    - Metadata storage
    """
    
    def __init__(self, version_dir: Path):
        """
        Initialize version manager.
        
        Parameters
        ----------
        version_dir : Path
            Directory for storing versions
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.version_dir / "versions.json"
        self.versions: Dict[str, DataVersion] = {}
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load version metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    self.versions = {
                        k: DataVersion.from_dict(v)
                        for k, v in data.items()
                    }
                print(f"[VERSION] Loaded {len(self.versions)} versions")
            except Exception as e:
                print(f"[VERSION] Error loading metadata: {e}")
    
    def _save_metadata(self):
        """Save version metadata to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.versions.items()}
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VERSION] Error saving metadata: {e}")
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for change detection."""
        # Use hash of data content
        content = df.to_csv()
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_version_id(self, source: str) -> str:
        """Generate version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{source}_{timestamp}"
    
    def save_version(
        self,
        df: pd.DataFrame,
        source: str,
        metadata: Dict = None
    ) -> DataVersion:
        """
        Save a new data version.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to version
        source : str
            Data source name
        metadata : dict, optional
            Additional metadata
        
        Returns
        -------
        DataVersion
            Version metadata
        """
        # Generate version ID
        version_id = self._generate_version_id(source)
        
        # Compute hash
        data_hash = self._compute_hash(df)
        
        # Check if data changed
        last_version = self.get_latest_version(source)
        if last_version and last_version.hash == data_hash:
            print(f"[VERSION] No changes detected for {source}")
            return last_version
        
        # Save data file
        data_file = self.version_dir / f"{version_id}.parquet"
        df.to_parquet(data_file)
        
        # Create version metadata
        version = DataVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            source=source,
            rows=len(df),
            columns=len(df.columns),
            hash=data_hash,
            start_date=str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else "",
            end_date=str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else "",
            metadata=metadata or {}
        )
        
        # Store version
        self.versions[version_id] = version
        self._save_metadata()
        
        print(f"[VERSION] Saved version: {version_id} ({len(df)} rows)")
        
        return version
    
    def load_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """
        Load a specific version.
        
        Parameters
        ----------
        version_id : str
            Version ID to load
        
        Returns
        -------
        pd.DataFrame or None
            Versioned data
        """
        if version_id not in self.versions:
            print(f"[VERSION] Version not found: {version_id}")
            return None
        
        data_file = self.version_dir / f"{version_id}.parquet"
        
        if not data_file.exists():
            print(f"[VERSION] Data file not found: {data_file}")
            return None
        
        try:
            df = pd.read_parquet(data_file)
            print(f"[VERSION] Loaded version: {version_id}")
            return df
        except Exception as e:
            print(f"[VERSION] Error loading version: {e}")
            return None
    
    def get_latest_version(self, source: str) -> Optional[DataVersion]:
        """
        Get latest version for a source.
        
        Parameters
        ----------
        source : str
            Data source name
        
        Returns
        -------
        DataVersion or None
            Latest version metadata
        """
        source_versions = [
            v for v in self.versions.values()
            if v.source == source
        ]
        
        if not source_versions:
            return None
        
        # Sort by timestamp
        source_versions.sort(key=lambda v: v.timestamp, reverse=True)
        return source_versions[0]
    
    def load_latest(self, source: str) -> Optional[pd.DataFrame]:
        """
        Load latest version for a source.
        
        Parameters
        ----------
        source : str
            Data source name
        
        Returns
        -------
        pd.DataFrame or None
            Latest versioned data
        """
        version = self.get_latest_version(source)
        if version:
            return self.load_version(version.version_id)
        return None
    
    def list_versions(self, source: Optional[str] = None) -> List[DataVersion]:
        """
        List all versions, optionally filtered by source.
        
        Parameters
        ----------
        source : str, optional
            Filter by source name
        
        Returns
        -------
        list
            List of DataVersion objects
        """
        versions = list(self.versions.values())
        
        if source:
            versions = [v for v in versions if v.source == source]
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        return versions
    
    def get_version_history(self, source: str) -> pd.DataFrame:
        """
        Get version history as DataFrame.
        
        Parameters
        ----------
        source : str
            Data source name
        
        Returns
        -------
        pd.DataFrame
            Version history
        """
        versions = self.list_versions(source)
        
        if not versions:
            return pd.DataFrame()
        
        data = [v.to_dict() for v in versions]
        df = pd.DataFrame(data)
        
        return df[["version_id", "timestamp", "rows", "columns", "start_date", "end_date", "hash"]]
    
    def cleanup_old_versions(self, source: str, keep_last_n: int = 10):
        """
        Delete old versions, keeping only the most recent N.
        
        Parameters
        ----------
        source : str
            Data source name
        keep_last_n : int
            Number of recent versions to keep
        """
        versions = self.list_versions(source)
        
        if len(versions) <= keep_last_n:
            print(f"[VERSION] No cleanup needed ({len(versions)} <= {keep_last_n})")
            return
        
        # Delete old versions
        to_delete = versions[keep_last_n:]
        
        for version in to_delete:
            # Delete data file
            data_file = self.version_dir / f"{version.version_id}.parquet"
            if data_file.exists():
                data_file.unlink()
            
            # Remove from metadata
            del self.versions[version.version_id]
        
        self._save_metadata()
        
        print(f"[VERSION] Cleaned up {len(to_delete)} old versions for {source}")

