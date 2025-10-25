"""Alerting system for pipeline monitoring."""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import json


class AlertManager:
    """
    Multi-channel alert manager.
    
    Supported channels:
    - Email (SMTP)
    - Slack webhook
    - File logging
    - Console
    """
    
    def __init__(
        self,
        email_config: Dict = None,
        slack_webhook: str = None,
        log_file: Path = None,
        console: bool = True
    ):
        """
        Initialize alert manager.
        
        Parameters
        ----------
        email_config : dict, optional
            Email configuration:
            {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your_email@gmail.com",
                "password": "your_password",
                "from_addr": "your_email@gmail.com",
                "to_addrs": ["recipient@example.com"],
            }
        slack_webhook : str, optional
            Slack webhook URL
        log_file : Path, optional
            Path to alert log file
        console : bool
            Print alerts to console
        """
        self.email_config = email_config
        self.slack_webhook = slack_webhook
        self.log_file = log_file
        self.console = console
        
        # Alert history
        self.alerts = []
        
        # Alert level configuration
        self.level_emoji = {
            "info": "[INFO]",
            "warning": "[WARN]",
            "error": "[ERROR]",
            "success": "[OK]",
        }
        
        print("[ALERT] Alert manager initialized")
        if email_config:
            print(f"[ALERT]   - Email: {email_config.get('from_addr')}")
        if slack_webhook:
            print(f"[ALERT]   - Slack: enabled")
        if log_file:
            print(f"[ALERT]   - Log file: {log_file}")
    
    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        source: str = None,
        metadata: Dict = None
    ):
        """
        Send alert through all configured channels.
        
        Parameters
        ----------
        level : str
            Alert level: 'info', 'warning', 'error', 'success'
        title : str
            Alert title
        message : str
            Alert message
        source : str, optional
            Data source name
        metadata : dict, optional
            Additional metadata
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "title": title,
            "message": message,
            "source": source,
            "metadata": metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Send through channels
        if self.console:
            self._send_console(alert)
        
        if self.log_file:
            self._send_log_file(alert)
        
        if self.email_config and level in ["error", "warning"]:
            self._send_email(alert)
        
        if self.slack_webhook:
            self._send_slack(alert)
    
    def _send_console(self, alert: dict):
        """Print alert to console."""
        emoji = self.level_emoji.get(alert["level"], "")
        level = alert["level"].upper()
        
        print(f"\n{emoji} [{level}] {alert['title']}")
        print(f"  {alert['message']}")
        if alert['source']:
            print(f"  Source: {alert['source']}")
        print()
    
    def _send_log_file(self, alert: dict):
        """Append alert to log file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert) + "\n")
        
        except Exception as e:
            print(f"[ALERT] Failed to write log: {e}")
    
    def _send_email(self, alert: dict):
        """Send alert via email."""
        if not self.email_config:
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.email_config["from_addr"]
            msg["To"] = ", ".join(self.email_config["to_addrs"])
            msg["Subject"] = f"[{alert['level'].upper()}] {alert['title']}"
            
            body = f"""
Alert Level: {alert['level']}
Timestamp: {alert['timestamp']}
Source: {alert.get('source', 'N/A')}

{alert['message']}

---
This is an automated alert from the data pipeline.
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            server = smtplib.SMTP(
                self.email_config["smtp_server"],
                self.email_config["smtp_port"]
            )
            server.starttls()
            server.login(
                self.email_config["username"],
                self.email_config["password"]
            )
            server.send_message(msg)
            server.quit()
            
            print(f"[ALERT] Email sent to {self.email_config['to_addrs']}")
        
        except Exception as e:
            print(f"[ALERT] Failed to send email: {e}")
    
    def _send_slack(self, alert: dict):
        """Send alert to Slack via webhook."""
        if not self.slack_webhook:
            return
        
        try:
            emoji = self.level_emoji.get(alert["level"], "")
            
            payload = {
                "text": f"{emoji} *{alert['title']}*",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} {alert['title']}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Level:*\n{alert['level']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Source:*\n{alert.get('source', 'N/A')}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:*\n{alert['timestamp']}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": alert['message']
                        }
                    }
                ]
            }
            
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            print(f"[ALERT] Slack notification sent")
        
        except Exception as e:
            print(f"[ALERT] Failed to send Slack notification: {e}")
    
    def get_recent_alerts(self, level: str = None, limit: int = 50) -> List[dict]:
        """
        Get recent alerts.
        
        Parameters
        ----------
        level : str, optional
            Filter by level
        limit : int
            Maximum number of alerts
        
        Returns
        -------
        list
            List of alerts
        """
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        
        return list(reversed(alerts[-limit:]))
    
    def get_alert_summary(self) -> dict:
        """
        Get alert statistics.
        
        Returns
        -------
        dict
            Alert summary
        """
        total = len(self.alerts)
        
        by_level = {}
        for alert in self.alerts:
            level = alert["level"]
            by_level[level] = by_level.get(level, 0) + 1
        
        by_source = {}
        for alert in self.alerts:
            source = alert.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            "total_alerts": total,
            "by_level": by_level,
            "by_source": by_source,
        }

