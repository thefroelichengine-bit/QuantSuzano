"""Manual upload system for data that cannot be scraped."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from .validator import DataValidator
from .versioning import DataVersionManager


class ManualUploadManager:
    """
    Manage manual data uploads for sources without APIs.
    
    Primary use case: Pulp prices (FOEX)
    
    Features:
    - CSV/Excel upload
    - Schema validation
    - Duplicate detection
    - Version tracking
    """
    
    def __init__(
        self,
        upload_dir: Path,
        version_manager: DataVersionManager,
        validator: DataValidator = None
    ):
        """
        Initialize upload manager.
        
        Parameters
        ----------
        upload_dir : Path
            Directory for manual uploads
        version_manager : DataVersionManager
            Version manager
        validator : DataValidator, optional
            Data validator
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.version_manager = version_manager
        self.validator = validator or DataValidator()
        
        print(f"[UPLOAD] Manual upload manager initialized")
        print(f"  Upload dir: {self.upload_dir}")
    
    def upload_file(
        self,
        file_path: Path,
        source_name: str,
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
        schema: dict = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Upload and process a manual data file.
        
        Parameters
        ----------
        file_path : Path
            Path to upload file (CSV or Excel)
        source_name : str
            Data source name
        date_column : str
            Name of date column
        date_format : str
            Date format string
        schema : dict, optional
            Expected schema: {column_name: dtype}
        validate : bool
            Run validation checks
        
        Returns
        -------
        pd.DataFrame
            Processed data
        """
        print(f"\n[UPLOAD] Processing file: {file_path}")
        print(f"  Source: {source_name}")
        
        # Read file
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        print(f"[UPLOAD] Read {len(df)} rows, {len(df.columns)} columns")
        
        # Validate schema
        if schema:
            missing_cols = set(schema.keys()) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Convert dtypes
            for col, dtype in schema.items():
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"[UPLOAD] Warning: Could not convert {col} to {dtype}: {e}")
        
        # Process date column
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
            df = df.set_index(date_column)
            df = df.sort_index()
            
            # Drop rows with invalid dates
            invalid_dates = df.index.isna().sum()
            if invalid_dates > 0:
                print(f"[UPLOAD] Warning: Dropping {invalid_dates} rows with invalid dates")
                df = df[~df.index.isna()]
        else:
            raise ValueError(f"Date column '{date_column}' not found")
        
        # Check for duplicates
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            print(f"[UPLOAD] Warning: Removing {dup_count} duplicate dates")
            df = df[~df.index.duplicated(keep='last')]
        
        # Validate
        if validate:
            validation = self.validator.validate(
                df,
                name=source_name,
                check_freshness=False  # Manual uploads may be historical
            )
            
            if not validation.passed:
                print(f"[UPLOAD] Validation issues detected")
                # Continue anyway for manual uploads
        
        # Save version
        self.version_manager.save_version(
            df,
            source=source_name,
            metadata={
                "upload_type": "manual",
                "upload_file": str(file_path),
                "upload_time": datetime.now().isoformat(),
            }
        )
        
        print(f"[UPLOAD] Successfully uploaded {len(df)} rows for {source_name}")
        
        return df
    
    def upload_pulp_prices(
        self,
        file_path: Path,
        price_type: str = "BHKP"
    ) -> pd.DataFrame:
        """
        Convenience method for uploading pulp prices.
        
        Expected columns:
        - date: Date (YYYY-MM-DD)
        - price: Price in USD/ton
        - type: Pulp type (BHKP, BSKP, etc.)
        
        Parameters
        ----------
        file_path : Path
            CSV/Excel file path
        price_type : str
            Pulp type to filter (default: BHKP - Hardwood)
        
        Returns
        -------
        pd.DataFrame
            Processed pulp prices
        """
        schema = {
            "date": "str",
            "price": "float",
        }
        
        df = self.upload_file(
            file_path,
            source_name="pulp_prices",
            date_column="date",
            date_format="%Y-%m-%d",
            schema=None,  # Don't enforce, be flexible
            validate=True
        )
        
        # Filter by type if column exists
        if "type" in df.columns:
            df = df[df["type"] == price_type].copy()
            print(f"[UPLOAD] Filtered to {price_type}: {len(df)} rows")
        
        # Rename columns
        if "price" in df.columns:
            df = df.rename(columns={"price": "pulp_usd"})
        
        # Select relevant columns
        if "pulp_usd" in df.columns:
            df = df[["pulp_usd"]]
        
        return df
    
    def create_template(
        self,
        source_name: str,
        columns: list,
        output_path: Path
    ):
        """
        Create a template file for manual uploads.
        
        Parameters
        ----------
        source_name : str
            Source name
        columns : list
            List of column names
        output_path : Path
            Output template file path
        """
        # Create empty DataFrame with columns
        df = pd.DataFrame(columns=columns)
        
        # Add example row
        example_row = {col: f"<{col}>" for col in columns}
        df = pd.concat([df, pd.DataFrame([example_row])], ignore_index=True)
        
        # Save template
        if output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == ".xlsx":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {output_path.suffix}")
        
        print(f"[UPLOAD] Created template: {output_path}")
    
    def list_uploaded_files(self) -> list:
        """
        List all files in upload directory.
        
        Returns
        -------
        list
            List of file paths
        """
        files = list(self.upload_dir.glob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in [".csv", ".xlsx", ".xls"]]
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

