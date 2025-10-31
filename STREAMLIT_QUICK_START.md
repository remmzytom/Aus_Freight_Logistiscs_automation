# Streamlit Cloud Quick Start Guide

## 🚀 Step-by-Step Deployment

### Step 1: Go to Streamlit Cloud
1. Open your browser
2. Go to: **https://streamlit.io/cloud**
3. Click **"Sign up"** or **"Get started"**

### Step 2: Sign Up with GitHub
1. Click **"Continue with GitHub"**
2. Authorize Streamlit Cloud to access your GitHub account
3. Allow access to repositories (or just your specific repository)

### Step 3: Deploy Your App
1. In Streamlit Cloud dashboard, click **"New app"** button (top right)
2. Fill in the deployment form:

   **Connect a repository:**
   - Select: `remmzytom/Aus_Freight_Logistiscs_automation`
   
   **Branch:**
   - Select: `main`
   
   **Main file path:**
   - Enter: `app.py`
   
   **App URL (optional):**
   - Leave default or customize: `aus-freight-logistics` (example)

3. Click **"Deploy"** button

### Step 4: Wait for Deployment
- Streamlit will install dependencies (takes 1-2 minutes)
- Watch the logs in real-time
- You'll see when it's done!

### Step 5: Access Your Dashboard
- Once deployed, you'll get a URL like:
  - `https://aus-freight-logistics.streamlit.app`
- Share this URL with stakeholders

## 📋 Important Notes

### Data Files
⚠️ **Important:** Your data files (`data/exports_cleaned.csv`) are NOT in Git (too large).

**Options:**
1. **Wait for GitHub Actions** - Data will be generated on the 1st of each month
2. **Run automation manually** - Trigger GitHub Actions manually first
3. **Add data download logic** - Modify `app.py` to download data if missing (future enhancement)

### Automatic Updates
✅ **Streamlit Cloud automatically:**
- Redeploys when you push to `main` branch
- Shows deployment logs
- Restarts on errors

### Troubleshooting

**If app shows "File not found" error:**
- This means data files haven't been generated yet
- Solution: Run GitHub Actions workflow manually first to generate data
- Then refresh Streamlit Cloud (it will pick up the new data)

**If app won't start:**
- Check the logs in Streamlit Cloud dashboard
- Verify all dependencies in `requirements.txt`
- Make sure `app.py` is in the root directory

## 🎉 That's It!

Once deployed, your dashboard will be:
- ✅ Publicly accessible
- ✅ Auto-updating when you push code
- ✅ Ready for stakeholder review

---

**Need help?** Check the full guide: `STREAMLIT_CLOUD_SETUP.md`

