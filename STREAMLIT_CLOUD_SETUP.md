# Streamlit Cloud Setup Guide

## Overview
This guide will help you deploy your Freight Logistics Dashboard to Streamlit Cloud for automatic updates and public access.

## Step 1: Push Code to GitHub Main Branch

**Important:** Streamlit Cloud works best with the `main` branch.

1. Make sure all your code is committed and pushed to GitHub:
```bash
git checkout dev
git add .
git commit -m "Ready for production deployment"
git push origin dev
```

2. When ready, merge to main (or push directly to main):
```bash
git checkout -b main  # or merge dev to main
git push origin main
```

## Step 2: Sign Up for Streamlit Cloud

1. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)**
2. Click **"Sign up"** or **"Get started"**
3. Sign up using your **GitHub account** (recommended)
4. Authorize Streamlit Cloud to access your GitHub repositories

## Step 3: Deploy Your App

1. In Streamlit Cloud dashboard, click **"New app"**
2. Fill in the following:

   **Repository:** `remmzytom/Aus_Freight_Logistiscs_automation`
   
   **Branch:** `main` (or `dev` if you haven't merged yet)
   
   **Main file path:** `app.py`
   
   **App URL:** (auto-generated or choose custom)

3. Click **"Deploy"**

## Step 4: Configure Settings (Optional)

1. Click on your deployed app
2. Go to **"Settings"** (gear icon)
3. Configure:
   - **Python version:** 3.9 (or 3.10)
   - **Secrets** (if needed for email notifications)

## Step 5: Automatic Updates

**Streamlit Cloud automatically:**
- Deploys new updates when you push to the main branch
- Restarts the app if it crashes
- Shows logs for debugging

## Important Notes

### Data Files
- **Large data files** (`data/exports_2024_2025.csv`, `data/exports_cleaned.csv`) are **NOT in GitHub** (too large)
- Streamlit Cloud will need to **download fresh data** when the app starts
- The dashboard will download data from the ABS website automatically

### Automation Integration
- GitHub Actions runs monthly automation (1st of every month)
- The automation generates updated data files
- Streamlit Cloud will use the latest data when users access the dashboard

### For Company Handover
1. Transfer repository ownership to company GitHub account
2. Update Streamlit Cloud to use company account
3. Share the Streamlit Cloud URL with stakeholders
4. Ensure company has access to:
   - GitHub repository
   - Streamlit Cloud account
   - Email notifications (if configured)

## Troubleshooting

**App won't start:**
- Check that `app.py` exists in the root directory
- Verify all dependencies are in `requirements.txt`
- Check logs in Streamlit Cloud dashboard

**Data not showing:**
- Verify data download works (check `2024_2025_extractor.py`)
- Ensure data files path is correct in `app.py`
- Check Streamlit Cloud logs for errors

**Need help?**
 Dynamic guide: https://docs.streamlit.io/streamlit-community-cloud/get-started

---

**Your Streamlit Cloud app will be live at:** `https://[your-app-name].streamlit.app`

