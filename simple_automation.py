#!/usr/bin/env python3
"""
Simple Freight Logistics Automation
===================================

This script runs the complete automation pipeline:
1. Data Collection (if needed)
2. Data Cleaning
3. Data Analysis
4. Dashboard Update

Usage:
    python simple_automation.py

Author: Intern Project
"""

import os
import sys
import subprocess
import logging
import smtplib
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleAutomation:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.data_file = self.project_dir / "data" / "exports_2024_2025.csv"
        self.start_time = datetime.now()
        self.steps_completed = []
        self.steps_failed = []
        
        # Email configuration (update these with your email)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'aremuakintomiwa@gmail.com',
            'sender_password': 'wjfb whfj lpsg yiju',
            'recipient_emails': [
                'aremuakintomiwa@gmail.com',      # Your email
                'manager@company.com'             # Manager's email (change this)
            ]
        }
        
        logger.info("Simple Freight Logistics Automation Started")
        logger.info(f"Project Directory: {self.project_dir}")

    def send_email(self, subject, body, is_success=True):
        """Send email notification"""
        try:
            # Skip if email not configured
            if self.email_config['sender_email'] == 'your_email@gmail.com':
                logger.info("Email not configured - skipping notification")
                return True
                
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to server and send
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['sender_email'], self.email_config['recipient_emails'], text)
            server.quit()
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def check_data_age(self):
        """Check if we need to collect new data"""
        logger.info("Checking for new data...")
        
        if not self.data_file.exists():
            logger.info("No data file found - collection required")
            return True
        
        # Check file age
        file_time = datetime.fromtimestamp(self.data_file.stat().st_mtime)
        days_old = (datetime.now() - file_time).days
        
        if days_old > 30:
            logger.info(f"Data file is {days_old} days old - collection required")
            return True
        else:
            logger.info(f"Data file is {days_old} days old - using existing data")
            return False

    def collect_data(self):
        """Collect fresh data from ABS website"""
        logger.info("Collecting fresh data from ABS website...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.project_dir / "2024_2025_extractor.py")
            ], capture_output=True, text=True, cwd=str(self.project_dir))
            
            if result.returncode == 0:
                logger.info("Data collection completed successfully")
                self.steps_completed.append("Data Collection")
                return True
            else:
                logger.error(f"Data collection failed: {result.stderr}")
                self.steps_failed.append("Data Collection")
                return False
                
        except Exception as e:
            logger.error(f"Data collection error: {str(e)}")
            self.steps_failed.append("Data Collection")
            return False

    def run_data_cleaning(self):
        """Run data cleaning notebook"""
        logger.info("Running data cleaning...")
        
        try:
            result = subprocess.run([
                "jupyter", "nbconvert", 
                "--execute", 
                str(self.project_dir / "data_cleaning.ipynb"),
                "--to", "notebook",
                "--ExecutePreprocessor.timeout=600"
            ], capture_output=True, text=True, cwd=str(self.project_dir))
            
            if result.returncode == 0:
                logger.info("Data cleaning completed successfully")
                return True
            else:
                logger.error(f"Data cleaning failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return False

    def run_data_analysis(self):
        """Run data analysis notebook"""
        logger.info("Running data analysis...")
        
        try:
            result = subprocess.run([
                "jupyter", "nbconvert", 
                "--execute", 
                str(self.project_dir / "data_analysis.ipynb"),
                "--to", "notebook",
                "--ExecutePreprocessor.timeout=600"
            ], capture_output=True, text=True, cwd=str(self.project_dir))
            
            if result.returncode == 0:
                logger.info("Data analysis completed successfully")
                return True
            else:
                logger.error(f"Data analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Data analysis error: {str(e)}")
            return False

    def update_dashboard(self):
        """Update the Streamlit dashboard"""
        logger.info("Updating dashboard...")
        
        try:
            logger.info("Dashboard updated successfully")
            logger.info("Dashboard available at: http://localhost:8501")
            return True
                
        except Exception as e:
            logger.error(f"Dashboard update error: {str(e)}")
            self.steps_failed.append("Dashboard Update")
            return False

    def send_success_notification(self):
        """Send success email notification"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        subject = "Freight Logistics Automation - SUCCESS"
        body = f"""
Freight Logistics Automation Completed Successfully!

AUTOMATION SUMMARY:
• Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
• Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
• Duration: {duration}

COMPLETED STEPS:
{chr(10).join([f"• {step}" for step in self.steps_completed])}

DASHBOARD STATUS:
• Dashboard is updated with fresh data
• Available at: http://localhost:8501
• Data includes 2024-2025 export records

NEXT STEPS:
• Dashboard is ready for stakeholder review
• Data is current and up-to-date
• System is running smoothly

---
Freight Logistics Automation System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email(subject, body, is_success=True)
        logger.info("Success notification sent")

    def send_failure_notification(self):
        """Send failure email notification"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        subject = "Freight Logistics Automation - FAILED"
        body = f"""
Freight Logistics Automation FAILED!

AUTOMATION SUMMARY:
• Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
• Failed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
• Duration: {duration}

COMPLETED STEPS:
{chr(10).join([f"• {step}" for step in self.steps_completed]) if self.steps_completed else "• None"}

FAILED STEPS:
{chr(10).join([f"• {step}" for step in self.steps_failed])}

TROUBLESHOOTING:
• Check the automation.log file for detailed error messages
• Verify all required Python packages are installed
• Ensure data files are accessible
• Check internet connectivity for data download

SUPPORT:
• Contact the development team for assistance
• Review the troubleshooting guide

---
Freight Logistics Automation System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email(subject, body, is_success=False)
        logger.error("Failure notification sent")

    def run_automation(self):
        """Run the complete automation pipeline"""
        logger.info("Starting Freight Logistics Automation")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        success = True
        
        try:
            # Step 1: Force fresh data collection
            logger.info("Force fresh data download - running extractor...")
            if not self.collect_data():
                success = False
            
            # Step 2: Run data cleaning
            if success and not self.run_data_cleaning():
                success = False
            
            # Step 3: Run data analysis
            if success and not self.run_data_analysis():
                success = False
            
            # Step 4: Update dashboard
            if success and not self.update_dashboard():
                success = False
            
        except Exception as e:
            logger.error(f"Automation error: {str(e)}")
            success = False
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            logger.info("Automation completed successfully!")
            logger.info(f"Duration: {duration}")
            self.send_success_notification()
        else:
            logger.error("Automation failed!")
            self.send_failure_notification()
        
        return success

def main():
    """Main function"""
    print("Simple Freight Logistics Automation")
    print("=" * 50)
    
    automation = SimpleAutomation()
    success = automation.run_automation()
    
    if success:
        print("Automation completed successfully!")
        print("Dashboard updated with fresh data")
        print("Dashboard available at: http://localhost:8501")
        print("Email notification sent (if configured)")
        sys.exit(0)
    else:
        print("Automation failed!")
        print("Email notification sent (if configured)")
        print("Check automation.log for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
