#!/usr/bin/env python3
"""
Monthly Scheduler for Freight Logistics Automation
=================================================

This script schedules the automation to run monthly.
It can be run as a Windows service or scheduled task.

Usage:
    python scheduler.py

Author: Intern Project
"""

import schedule
import time
import subprocess
import sys
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_automation():
    """Run the automation pipeline"""
    logger.info("Starting scheduled automation...")
    
    try:
        # Run the simple automation script
        result = subprocess.run([
            sys.executable, 
            "simple_automation.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Scheduled automation completed successfully")
        else:
            logger.error(f"Scheduled automation failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Scheduled automation error: {str(e)}")

def main():
    """Main scheduler function"""
    logger.info("Freight Logistics Scheduler Started")
    
    # Schedule automation to run on the 1st of every month at 6 AM
    schedule.every().month.do(run_automation)
    
    # Also schedule a test run every day at 2 PM (for testing)
    schedule.every().day.at("14:00").do(run_automation)
    
    logger.info("Scheduler configured:")
    logger.info("- Monthly automation: 1st of every month at 6 AM")
    logger.info("- Daily test run: Every day at 2 PM")
    logger.info("Press Ctrl+C to stop the scheduler")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")

if __name__ == "__main__":
    main()
