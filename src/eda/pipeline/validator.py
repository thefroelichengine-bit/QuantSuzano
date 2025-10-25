"""Data validation module for quality checks."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    check_func: callable
    severity: str = "error"  # 'error', 'warning', 'info'
    description: str = ""


@dataclass
class ValidationResult:
    """Result of validation checks."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_issue(self, severity: str, message: str):
        """Add validation issue."""
        if severity == "error":
            self.errors.append(message)
            self.passed = False
        elif severity == "warning":
            self.warnings.append(message)
        elif severity == "info":
            self.info.append(message)
    
    def summary(self) -> str:
        """Get summary string."""
        lines = []
        if not self.passed:
            lines.append(f"[VALIDATION FAILED] {len(self.errors)} errors")
        else:
            lines.append("[VALIDATION PASSED]")
        
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        
        if self.info:
            lines.append(f"Info ({len(self.info)}):")
            for inf in self.info:
                lines.append(f"  - {inf}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Comprehensive data validator for time series data.
    
    Checks:
    - Completeness (missing dates, NaN values)
    - Consistency (data types, ranges)
    - Quality (outliers, duplicates)
    - Freshness (data recency)
    """
    
    def __init__(self):
        """Initialize validator with default rules."""
        self.rules: List[ValidationRule] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        # Completeness checks
        self.add_rule(
            "no_empty_dataframe",
            lambda df: len(df) > 0,
            severity="error",
            description="DataFrame must not be empty"
        )
        
        self.add_rule(
            "has_datetime_index",
            lambda df: isinstance(df.index, pd.DatetimeIndex),
            severity="error",
            description="Index must be DatetimeIndex"
        )
        
        self.add_rule(
            "sorted_index",
            lambda df: df.index.is_monotonic_increasing,
            severity="warning",
            description="Index should be sorted chronologically"
        )
    
    def add_rule(
        self,
        name: str,
        check_func: callable,
        severity: str = "error",
        description: str = ""
    ):
        """Add custom validation rule."""
        rule = ValidationRule(
            name=name,
            check_func=check_func,
            severity=severity,
            description=description
        )
        self.rules.append(rule)
    
    def validate(
        self,
        df: pd.DataFrame,
        name: str = "data",
        check_missing: bool = True,
        check_outliers: bool = True,
        check_duplicates: bool = True,
        check_freshness: bool = True,
        freshness_days: int = 7,
        outlier_std: float = 5.0,
    ) -> ValidationResult:
        """
        Run all validation checks.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        name : str
            Data name for messages
        check_missing : bool
            Check for missing values
        check_outliers : bool
            Check for outliers
        check_duplicates : bool
            Check for duplicate rows
        check_freshness : bool
            Check data freshness
        freshness_days : int
            Maximum age in days for fresh data
        outlier_std : float
            Number of standard deviations for outlier detection
        
        Returns
        -------
        ValidationResult
            Validation results
        """
        result = ValidationResult(passed=True)
        
        print(f"\n[VALIDATOR] Validating {name}...")
        
        # Run registered rules
        for rule in self.rules:
            try:
                if not rule.check_func(df):
                    result.add_issue(
                        rule.severity,
                        f"{rule.name}: {rule.description}"
                    )
            except Exception as e:
                result.add_issue("error", f"{rule.name} failed: {e}")
        
        # Missing values check
        if check_missing:
            self._check_missing(df, result)
        
        # Outliers check
        if check_outliers:
            self._check_outliers(df, result, outlier_std)
        
        # Duplicates check
        if check_duplicates:
            self._check_duplicates(df, result)
        
        # Freshness check
        if check_freshness:
            self._check_freshness(df, result, freshness_days)
        
        print(result.summary())
        
        return result
    
    def _check_missing(self, df: pd.DataFrame, result: ValidationResult):
        """Check for missing values."""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        
        if missing_cells > 0:
            missing_pct = 100 * missing_cells / total_cells
            
            # Per-column missing
            missing_per_col = df.isna().sum()
            missing_cols = missing_per_col[missing_per_col > 0]
            
            severity = "error" if missing_pct > 10 else "warning"
            
            msg = f"Missing values: {missing_cells}/{total_cells} ({missing_pct:.1f}%)"
            if len(missing_cols) > 0:
                msg += f" in columns: {dict(missing_cols)}"
            
            result.add_issue(severity, msg)
        else:
            result.add_issue("info", "No missing values")
    
    def _check_outliers(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        outlier_std: float
    ):
        """Check for statistical outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            outliers = np.abs(df[col] - mean) > outlier_std * std
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_pct = 100 * n_outliers / len(df)
                severity = "error" if outlier_pct > 5 else "warning"
                
                msg = f"Outliers in {col}: {n_outliers} ({outlier_pct:.1f}%) beyond {outlier_std}σ"
                result.add_issue(severity, msg)
    
    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult):
        """Check for duplicate rows."""
        # Duplicate indices
        dup_index = df.index.duplicated()
        n_dup_index = dup_index.sum()
        
        if n_dup_index > 0:
            result.add_issue(
                "error",
                f"Duplicate index values: {n_dup_index}"
            )
        
        # Duplicate rows
        dup_rows = df.duplicated()
        n_dup_rows = dup_rows.sum()
        
        if n_dup_rows > 0:
            result.add_issue(
                "warning",
                f"Duplicate rows: {n_dup_rows}"
            )
    
    def _check_freshness(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        freshness_days: int
    ):
        """Check data freshness."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        last_date = df.index.max()
        age_days = (datetime.now() - last_date).days
        
        if age_days > freshness_days:
            severity = "error" if age_days > 30 else "warning"
            result.add_issue(
                severity,
                f"Data is {age_days} days old (last: {last_date.date()})"
            )
        else:
            result.add_issue("info", f"Data is fresh (last: {last_date.date()}, {age_days} days old)")
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        name: str = "data"
    ) -> ValidationResult:
        """
        Validate DataFrame schema.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        expected_columns : list
            Expected column names
        name : str
            Data name
        
        Returns
        -------
        ValidationResult
            Validation results
        """
        result = ValidationResult(passed=True)
        
        print(f"\n[VALIDATOR] Validating schema for {name}...")
        
        actual_cols = set(df.columns)
        expected_cols = set(expected_columns)
        
        # Missing columns
        missing = expected_cols - actual_cols
        if missing:
            result.add_issue("error", f"Missing columns: {missing}")
        
        # Extra columns
        extra = actual_cols - expected_cols
        if extra:
            result.add_issue("info", f"Extra columns: {extra}")
        
        print(result.summary())
        
        return result

