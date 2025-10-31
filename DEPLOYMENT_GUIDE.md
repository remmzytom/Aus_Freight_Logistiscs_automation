# Complete Deployment Guide

## Project Overview
This project automates monthly freight logistics data collection, cleaning, analysis, and dashboard updates.

## Architecture

### Components
1. **Data Collection** (`2024_2025_extractor.py`) - Downloads 2024-2025 data from ABS
2. **Data Cleaning** (`data_cleaning.ipynb`) - Cleans and processes raw data
3. **Data Analysis** (`data_analysis.ipynb`) - Generates insights and visualizations
4. **Dashboard** (`app.py`) - Streamlit interactive dashboard
5. **Automation** (`simple_automation.py`) - Orchestrates the entire pipeline

### Automation Flow
```
Monthly Schedule (1st of month, 6 AM UTC)
    ↓
GitHub Actions Workflow
    ↓
1. Data Collection (download from ABS)
    ↓
2. Data Cleaning (process raw data)
    ↓
3. Data Analysis (generate insights)
    ↓
4. Email Notification (success/failure)
    ↓
5. Dashboard Auto-Update (Streamlit Cloud)
```

## Prerequisites

### Required Accounts
1. **GitHub Account** - For code repository and GitHub Actions
2. **Streamlit Cloud Account** - For dashboard hosting (free)
3. **Email Account** - For notifications (Gmail recommended)

### Required Software (Local Development)
- Python 3.9+
- Git
- All packages in `requirements.txt`

## Setup Steps

### 1. GitHub Repository Setup

**Current Status:**
- Repository: `https://github.com/remmzytom/Aus_Freight_Logistiscs_automation.git`
- Working Branch: `dev`
- Main Branch: (to be created when ready)

**To deploy:**
```bash
# Merge dev to main when ready
git checkout dev
git pull origin dev
git checkout -b main
git push origin main
```

### 2. GitHub Actions Setup

**Location:** `.github/workflows/monthly_automation.yml`

**Current Schedule:**
- Runs: 1st of every month at 6 AM UTC
- Manual trigger: Available in GitHub Actions tab

**To test manually:**
1. Go to your GitHub repository
2. Click **"Actions"** tab
3. Select **"Monthly Freight Data Automation"**
4. Click **"Run workflow"** button

**Email Configuration (Optional):**
Add GitHub Secrets for email notifications:
- Go to Repository → Settings → Secrets and variables → Actions
- Add secrets:
  - `SENDER_EMAIL`: Your email
  - `SENDER_PASSWORD`: App password (Gmail)
  - `RECIPIENT_EMAILS`: Comma-separated recipient emails

### 3. Streamlit Cloud Setup

**See:** `STREAMLIT_CLOUD_SETUP.md` for detailed instructions

**Quick Steps:**
1. Sign up at https://streamlit.io/cloud (using GitHub)
2. Click **"New app"**
3. Select your repository: `remmzytom/Aus_Freight_Logistiscs_automation`
4. Branch: `main`
5. Main file: `app.py`
6. Click **"Deploy"**

**Note:** Streamlit Cloud automatically redeploys when you push to main branch.

### 4. Local Scheduler (Optional)

If you want to run automation locally instead of GitHub Actions:

```bash
python scheduler.py
```

This runs automation once per month (on the day you start it).

## Testing

### Test Automation Locally
```bash
python simple_automation.py
```

### Test Dashboard Locally
```bash
streamlit run app.py
```

### Test GitHub Actions
- Go to GitHub Actions tab
- Click "Run workflow" to trigger manually

## Monitoring

### Success Indicators
- GitHub Actions shows green checkmark after monthly run
- Email notification received (if configured)
- Dashboard shows updated data with recent timestamp
- Streamlit Cloud shows app is running

### Failure Indicators
- GitHub Actions shows red X
- Email notification with failure details
- Dashboard shows old data
- Check `automation.log` for details

### Logs
- **GitHub Actions:** View in Actions tab → specific workflow run
- **Local:** Check `automation.log` file
- **Streamlit Cloud:** View in app settings → logs

## Maintenance

### Monthly Tasks
- Automation runs automatically (no action needed)
- Check email notifications for success/failure
- Verify dashboard is showing updated data

### Troubleshooting

**Automation Fails:**
1. Check GitHub Actions logs
2. Verify ABS website is accessible
3. Check if data format changed
4. Review error messages in logs соп

**Dashboard Not Updating:**
1. Verify Streamlit Cloud deployment is active
2. Check if main branch has latest code
3. Verify data files are accessible
4. Check Streamlit Cloud logs

**Email Notifications Not Working:**
1. Verify email credentials in GitHub Secrets
2. Check Gmail app password is correct
3. Verify recipient emails are valid
4. Check spam folder

## For Company Handover

### What to Transfer
1. **GitHub Repository Access**
   - Transfer ownership or add company as collaborator
   - Ensure company has admin access

2. **Streamlit Cloud Access**
   - Add company account as collaborator
   - Or transfer app ownership

3. **Email Configuration**
   - Update recipient emails to company addresses
   - Update sender email if needed

4. **Documentation**
   - Share this deployment guide
   - Share `STREAMLIT_CLOUD_SETUP.md`
   - Share `EMAIL_SETUP.md`

### Knowledge Transfer
- Explain automation flow
- Show how to monitor GitHub Actions
- Demonstrate dashboard features
- Explain troubleshooting steps

## Support Resources

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Streamlit Docs:** https://docs.streamlit.io

---

**Last Updated:** October 2025
**Maintained By:** Intern Project Team

