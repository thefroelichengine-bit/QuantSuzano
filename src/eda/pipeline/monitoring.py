"""Monitoring and dashboard for pipeline status."""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from .orchestrator import DataPipeline
from .scheduler import DataScheduler
from .alerting import AlertManager


class PipelineMonitor:
    """
    Monitor pipeline health and generate status reports.
    
    Provides:
    - Data freshness checks
    - Error rate monitoring
    - Version history tracking
    - Health scores
    """
    
    def __init__(
        self,
        pipeline: DataPipeline,
        scheduler: DataScheduler = None,
        alert_manager: AlertManager = None
    ):
        """
        Initialize monitor.
        
        Parameters
        ----------
        pipeline : DataPipeline
            Pipeline instance
        scheduler : DataScheduler, optional
            Scheduler instance
        alert_manager : AlertManager, optional
            Alert manager instance
        """
        self.pipeline = pipeline
        self.scheduler = scheduler
        self.alert_manager = alert_manager
    
    def check_data_freshness(self, max_age_hours: int = 48) -> Dict:
        """
        Check freshness of all data sources.
        
        Parameters
        ----------
        max_age_hours : int
            Maximum acceptable age in hours
        
        Returns
        -------
        dict
            Freshness status for each source
        """
        status = {}
        
        for source_name in self.pipeline.SOURCES.keys():
            version = self.pipeline.version_manager.get_latest_version(source_name)
            
            if version is None:
                status[source_name] = {
                    "status": "missing",
                    "age_hours": None,
                    "last_update": None,
                }
            else:
                last_update = datetime.fromisoformat(version.timestamp)
                age = datetime.now() - last_update
                age_hours = age.total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    health = "stale"
                else:
                    health = "fresh"
                
                status[source_name] = {
                    "status": health,
                    "age_hours": age_hours,
                    "last_update": last_update.isoformat(),
                    "rows": version.rows,
                }
        
        return status
    
    def get_health_score(self) -> Dict:
        """
        Compute overall pipeline health score.
        
        Returns
        -------
        dict
            Health score and metrics
        """
        freshness = self.check_data_freshness()
        
        # Count sources by status
        total_sources = len(freshness)
        fresh_sources = sum(1 for s in freshness.values() if s["status"] == "fresh")
        stale_sources = sum(1 for s in freshness.values() if s["status"] == "stale")
        missing_sources = sum(1 for s in freshness.values() if s["status"] == "missing")
        
        # Compute health score (0-100)
        if total_sources == 0:
            score = 0
        else:
            score = (fresh_sources / total_sources) * 100
        
        # Determine overall status
        if score >= 80:
            overall_status = "healthy"
        elif score >= 50:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "score": score,
            "status": overall_status,
            "total_sources": total_sources,
            "fresh": fresh_sources,
            "stale": stale_sources,
            "missing": missing_sources,
        }
    
    def get_error_rate(self, hours: int = 24) -> Dict:
        """
        Calculate error rate over time period.
        
        Parameters
        ----------
        hours : int
            Time period in hours
        
        Returns
        -------
        dict
            Error rate metrics
        """
        if not self.scheduler:
            return {"error": "No scheduler configured"}
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        history = self.scheduler.history
        recent = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]
        
        if not recent:
            return {
                "total_runs": 0,
                "errors": 0,
                "error_rate": 0,
            }
        
        total = len(recent)
        errors = sum(1 for h in recent if not h["success"])
        error_rate = errors / total if total > 0 else 0
        
        return {
            "total_runs": total,
            "errors": errors,
            "error_rate": error_rate,
            "period_hours": hours,
        }
    
    def generate_status_report(self) -> str:
        """
        Generate comprehensive status report.
        
        Returns
        -------
        str
            Formatted status report
        """
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("DATA PIPELINE STATUS REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        lines.append("")
        
        # Health score
        health = self.get_health_score()
        lines.append(f"OVERALL HEALTH: {health['status'].upper()} (Score: {health['score']:.1f}/100)")
        lines.append(f"  Fresh sources:   {health['fresh']}/{health['total_sources']}")
        lines.append(f"  Stale sources:   {health['stale']}/{health['total_sources']}")
        lines.append(f"  Missing sources: {health['missing']}/{health['total_sources']}")
        lines.append("")
        
        # Data freshness
        lines.append("DATA FRESHNESS:")
        freshness = self.check_data_freshness()
        
        for source, info in freshness.items():
            status_icon = {
                "fresh": "✓",
                "stale": "⚠",
                "missing": "✗"
            }.get(info["status"], "?")
            
            if info["age_hours"] is None:
                age_str = "N/A"
            else:
                age_str = f"{info['age_hours']:.1f}h"
            
            lines.append(f"  {status_icon} {source:20s} {info['status']:10s} (age: {age_str})")
        
        lines.append("")
        
        # Error rate
        if self.scheduler:
            error_rate = self.get_error_rate(hours=24)
            lines.append("ERROR RATE (24h):")
            lines.append(f"  Total runs: {error_rate['total_runs']}")
            lines.append(f"  Errors:     {error_rate['errors']}")
            lines.append(f"  Error rate: {error_rate['error_rate']*100:.1f}%")
            lines.append("")
        
        # Recent alerts
        if self.alert_manager:
            recent_alerts = self.alert_manager.get_recent_alerts(limit=5)
            lines.append("RECENT ALERTS (last 5):")
            
            if recent_alerts:
                for alert in recent_alerts:
                    timestamp = datetime.fromisoformat(alert["timestamp"])
                    lines.append(f"  [{alert['level']:7s}] {timestamp.strftime('%Y-%m-%d %H:%M')} - {alert['title']}")
            else:
                lines.append("  No recent alerts")
            
            lines.append("")
        
        # Version history
        lines.append("VERSION COUNTS:")
        for source in self.pipeline.SOURCES.keys():
            versions = self.pipeline.version_manager.list_versions(source)
            lines.append(f"  {source:20s} {len(versions)} versions")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def export_metrics_csv(self, output_file: Path):
        """
        Export monitoring metrics to CSV.
        
        Parameters
        ----------
        output_file : Path
            Output CSV file
        """
        freshness = self.check_data_freshness()
        
        data = []
        for source, info in freshness.items():
            data.append({
                "source": source,
                "status": info["status"],
                "age_hours": info["age_hours"],
                "last_update": info.get("last_update"),
                "rows": info.get("rows"),
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"[MONITOR] Exported metrics to {output_file}")
    
    def run_health_check(self, alert_on_issues: bool = True) -> bool:
        """
        Run health check and optionally send alerts.
        
        Parameters
        ----------
        alert_on_issues : bool
            Send alerts if issues detected
        
        Returns
        -------
        bool
            True if healthy, False if issues detected
        """
        print("\n[MONITOR] Running health check...")
        
        health = self.get_health_score()
        freshness = self.check_data_freshness()
        
        issues = []
        
        # Check for missing sources
        for source, info in freshness.items():
            if info["status"] == "missing":
                issues.append(f"Missing data: {source}")
        
        # Check for stale sources
        for source, info in freshness.items():
            if info["status"] == "stale":
                issues.append(f"Stale data: {source} ({info['age_hours']:.1f}h old)")
        
        # Report
        if issues:
            print(f"[MONITOR] Health check FAILED ({len(issues)} issues)")
            for issue in issues:
                print(f"  - {issue}")
            
            if alert_on_issues and self.alert_manager:
                self.alert_manager.send_alert(
                    level="warning",
                    title="Pipeline health check failed",
                    message="\n".join(issues),
                )
            
            return False
        else:
            print("[MONITOR] Health check PASSED")
            return True

