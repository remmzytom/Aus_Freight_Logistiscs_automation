# GitHub Actions Auto-Deployment Setup

## Overview

This workflow automatically deploys your Streamlit app to Google Cloud Run whenever you push changes to the `main` branch.

## How It Works

```
Push to GitHub (main branch)
    ↓
GitHub Actions triggers
    ↓
Builds Docker image
    ↓
Pushes to Google Container Registry
    ↓
Deploys to Cloud Run
    ↓
Your dashboard is updated automatically! 🎉
```

## Setup Steps

### Step 1: Create Service Account in Google Cloud

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Click **"Create Service Account"**
4. Name: `github-actions-deployer`
5. Description: `Service account for GitHub Actions deployment`
6. Click **"Create and Continue"**

### Step 2: Grant Permissions

Add these roles to the service account:
- **Cloud Run Admin** (`roles/run.admin`)
- **Service Account User** (`roles/iam.serviceAccountUser`)
- **Storage Admin** (`roles/storage.admin`) - for pushing Docker images
- **Artifact Registry Writer** (`roles/artifactregistry.writer`) - if using Artifact Registry

**Steps:**
1. In Service Accounts, click on `github-actions-deployer`
2. Click **"Permissions"** tab
3. Click **"Grant Access"**
4. Add the roles above
5. Click **"Save"**

### Step 3: Create Service Account Key

1. Still in the service account page
2. Click **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"**
4. Select **JSON** format
5. Click **"Create"**
6. **IMPORTANT:** Save the downloaded JSON file securely

### Step 4: Add Secret to GitHub

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `GCP_SA_KEY`
5. Value: Paste the **entire contents** of the JSON file you downloaded
6. Click **"Add secret"**

### Step 5: Enable Required APIs

Run these commands in Google Cloud Shell or your terminal:

```bash
gcloud config set project manifest-wind-478914-p8

gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com
```

### Step 6: Push the Workflow File

The workflow file is already created at `.github/workflows/deploy-cloud-run.yml`. Just commit and push:

```bash
git add .github/workflows/deploy-cloud-run.yml
git commit -m "Add GitHub Actions workflow for auto-deployment to Cloud Run"
git push
```

## Testing

### Test Automatic Deployment

1. Make a small change to `app.py` (e.g., add a comment)
2. Commit and push:
   ```bash
   git add app.py
   git commit -m "Test auto-deployment"
   git push
   ```
3. Go to **Actions** tab in GitHub
4. Watch the workflow run
5. Check your Cloud Run dashboard - it should update automatically!

### Test Manual Deployment

1. Go to **Actions** tab
2. Click **"Deploy to Cloud Run"**
3. Click **"Run workflow"** → **"Run workflow"**

## Workflow Triggers

The workflow runs automatically when:
- ✅ Code is pushed to `main` branch
- ✅ Files changed: `app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`
- ✅ Manual trigger from Actions tab

## Monitoring

### View Deployment Status:
1. Go to **Actions** tab in GitHub
2. Click on **"Deploy to Cloud Run"** workflow
3. See deployment logs and status

### View Cloud Run Logs:
```bash
gcloud run services logs read aus-freight-dashboard --region us-central1
```

## Troubleshooting

### Workflow fails with "Permission denied"
- Check service account has correct roles
- Verify `GCP_SA_KEY` secret is set correctly
- Ensure JSON key file is complete

### Docker build fails
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Check workflow logs for specific errors

### Deployment fails
- Verify Cloud Run API is enabled
- Check service account permissions
- Review Cloud Run logs

### Image push fails
- Check Container Registry API is enabled
- Verify service account has Storage Admin role
- Check Docker authentication

## Cost

**GitHub Actions:**
- 2,000 minutes/month free
- Your usage: ~5-10 minutes per deployment
- **Cost: FREE** ✅

**Cloud Run:**
- Same as manual deployment
- No additional cost for auto-deployment

## Benefits

✅ **Automatic** - No manual deployment needed  
✅ **Fast** - Deploys in ~5-10 minutes  
✅ **Reliable** - Runs on GitHub's infrastructure  
✅ **Free** - Uses GitHub Actions free tier  
✅ **Safe** - Only deploys when code changes  

## Next Steps

After setup:
1. ✅ Push a test change to verify deployment
2. ✅ Check Cloud Run dashboard updates automatically
3. ✅ Monitor deployment in Actions tab
4. ✅ Enjoy automatic deployments! 🎉

Your dashboard will now update automatically every time you push changes to GitHub!

