"""Scheduler for automated data updates."""

import time
import schedule
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict
from .orchestrator import DataPipeline
from .alerting import AlertManager


class DataScheduler:
    """
    Schedule automated data updates.
    
    Features:
    - Cron-like scheduling
    - Per-source update frequencies
    - Error handling and alerts
    - Status monitoring
    """
    
    def __init__(
        self,
        pipeline: DataPipeline,
        alert_manager: AlertManager = None
    ):
        """
        Initialize scheduler.
        
        Parameters
        ----------
        pipeline : DataPipeline
            Data pipeline instance
        alert_manager : AlertManager, optional
            Alert manager for notifications
        """
        self.pipeline = pipeline
        self.alert_manager = alert_manager
        
        self.jobs = []
        self.is_running = False
        
        # Execution history
        self.history = []
        
        print("[SCHEDULER] Initialized")
    
    def schedule_source(
        self,
        source_name: str,
        interval_hours: int = None
    ):
        """
        Schedule updates for a specific source.
        
        Parameters
        ----------
        source_name : str
            Data source name
        interval_hours : int, optional
            Update interval in hours (default: from config)
        """
        if source_name not in self.pipeline.SOURCES:
            raise ValueError(f"Unknown source: {source_name}")
        
        config = self.pipeline.SOURCES[source_name]
        
        if interval_hours is None:
            interval_hours = config["update_frequency_hours"]
        
        def job():
            """Job wrapper with error handling and alerts."""
            print(f"\n[SCHEDULER] Running scheduled update: {source_name}")
            start_time = datetime.now()
            
            try:
                df = self.pipeline.fetch_source(source_name, force_full=False)
                
                if df is not None:
                    success = True
                    error = None
                    print(f"[SCHEDULER] Update successful: {source_name}")
                else:
                    success = False
                    error = "No data returned"
                    print(f"[SCHEDULER] Update failed: {source_name}")
            
            except Exception as e:
                success = False
                error = str(e)
                print(f"[SCHEDULER] Update error: {source_name} - {error}")
            
            # Record history
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.history.append({
                "source": source_name,
                "timestamp": start_time.isoformat(),
                "success": success,
                "error": error,
                "duration_seconds": duration,
            })
            
            # Send alert if failed
            if not success and self.alert_manager:
                self.alert_manager.send_alert(
                    level="error",
                    title=f"Data update failed: {source_name}",
                    message=f"Error: {error}\nDuration: {duration:.1f}s",
                    source=source_name
                )
            
            # Send success notification for important sources
            elif success and config.get("required", False) and self.alert_manager:
                self.alert_manager.send_alert(
                    level="info",
                    title=f"Data update successful: {source_name}",
                    message=f"Duration: {duration:.1f}s",
                    source=source_name
                )
        
        # Schedule the job
        schedule.every(interval_hours).hours.do(job)
        
        self.jobs.append({
            "source": source_name,
            "interval_hours": interval_hours,
            "job": job,
        })
        
        print(f"[SCHEDULER] Scheduled {source_name} every {interval_hours}h")
    
    def schedule_all(self):
        """Schedule updates for all configured sources."""
        print("[SCHEDULER] Scheduling all sources")
        
        for source_name in self.pipeline.SOURCES.keys():
            self.schedule_source(source_name)
        
        print(f"[SCHEDULER] Scheduled {len(self.jobs)} sources")
    
    def run_once(self):
        """Run all scheduled jobs once immediately."""
        print("[SCHEDULER] Running all jobs once")
        
        for job in self.jobs:
            job["job"]()
    
    def start(self, run_immediately: bool = True):
        """
        Start the scheduler.
        
        Parameters
        ----------
        run_immediately : bool
            Run all jobs immediately before starting schedule
        """
        print("[SCHEDULER] Starting scheduler")
        
        self.is_running = True
        
        # Run immediately if requested
        if run_immediately:
            self.run_once()
        
        # Main scheduler loop
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            print("\n[SCHEDULER] Stopped by user")
            self.stop()
    
    def stop(self):
        """Stop the scheduler."""
        print("[SCHEDULER] Stopping scheduler")
        self.is_running = False
    
    def get_status(self) -> dict:
        """
        Get scheduler status.
        
        Returns
        -------
        dict
            Status information
        """
        return {
            "is_running": self.is_running,
            "jobs_count": len(self.jobs),
            "history_count": len(self.history),
            "jobs": [
                {
                    "source": job["source"],
                    "interval_hours": job["interval_hours"],
                }
                for job in self.jobs
            ]
        }
    
    def get_history(self, source: str = None, limit: int = 100) -> List[dict]:
        """
        Get execution history.
        
        Parameters
        ----------
        source : str, optional
            Filter by source name
        limit : int
            Maximum number of records
        
        Returns
        -------
        list
            List of execution records
        """
        history = self.history
        
        if source:
            history = [h for h in history if h["source"] == source]
        
        # Return most recent first
        return list(reversed(history[-limit:]))
    
    def export_schedule_cron(self, output_file: Path):
        """
        Export schedule as cron-compatible script.
        
        Parameters
        ----------
        output_file : Path
            Output file path
        """
        lines = [
            "#!/bin/bash",
            "# Auto-generated cron script for data pipeline",
            "# Generated: " + datetime.now().isoformat(),
            "",
        ]
        
        for job in self.jobs:
            source = job["source"]
            hours = job["interval_hours"]
            
            # Convert hours to cron format
            if hours == 1:
                cron = "0 * * * *"  # Every hour
            elif hours == 24:
                cron = "0 0 * * *"  # Daily at midnight
            elif hours == 168:
                cron = "0 0 * * 0"  # Weekly on Sunday
            else:
                cron = f"0 */{hours} * * *"  # Every N hours
            
            lines.append(f"# {source} (every {hours}h)")
            lines.append(f"{cron} cd $(dirname $0) && python -m eda.cli pipeline-run --sources {source}")
            lines.append("")
        
        output_file.write_text("\n".join(lines))
        print(f"[SCHEDULER] Exported cron script to {output_file}")

