# Google Cloud Run Deployment Guide

This guide will walk you through deploying your Streamlit dashboard to Google Cloud Run manually.

## Prerequisites

- Google account (Gmail account works)
- Credit card (for billing, but free tier available)
- Docker Desktop installed (for building the image)
- Terminal/Command Prompt access

## Step 1: Create Google Cloud Account

1. Go to https://console.cloud.google.com/
2. Sign in with your Google account
3. Accept terms and conditions
4. **Create a new project:**
   - Click "Select a project" → "New Project"
   - Project name: `aus-freight-dashboard` (or any name)
   - Note your **Project ID** (you'll need this later)
   - Click "Create"

## Step 2: Enable Billing

1. In Google Cloud Console, go to "Billing"
2. Link a billing account (credit card required)
3. **Don't worry** - Google gives $300 free credit for new accounts
4. Free tier covers 2 million requests/month

## Step 3: Install Google Cloud SDK (gcloud CLI)

### Windows:
1. Download installer: https://cloud.google.com/sdk/docs/install
2. Run the installer
3. Follow the installation wizard
4. Restart your terminal

### Mac:
```bash
# Using Homebrew
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### Linux:
```bash
# Download and install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## Step 4: Initialize gcloud

1. Open terminal/command prompt
2. Run:
```bash
gcloud init
```

3. Follow the prompts:
   - Login with your Google account
   - Select your project (the one you created)
   - Choose default region (e.g., `us-central1`)

## Step 5: Install Docker Desktop

1. Download: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Verify it's running (Docker icon in system tray)

## Step 6: Configure Deployment Script

1. Open `deploy-cloud-run.sh` (or create `deploy-cloud-run.bat` for Windows)
2. Update these values:
   ```bash
   PROJECT_ID="your-project-id"  # Your GCP Project ID
   REGION="us-central1"          # Your preferred region
   ```

## Step 7: Deploy to Cloud Run

### For Windows (PowerShell):
Create `deploy-cloud-run.ps1` or use the commands manually:

```powershell
# Set your project ID
$PROJECT_ID = "your-project-id"
$SERVICE_NAME = "aus-freight-dashboard"
$REGION = "us-central1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Set project
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push image
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --memory 8Gi `
    --cpu 4 `
    --timeout 600 `
    --min-instances 0 `
    --max-instances 10 `
    --port 8080

# Get URL
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

### For Mac/Linux:
```bash
# Make script executable
chmod +x deploy-cloud-run.sh

# Update PROJECT_ID in the script first, then run:
./deploy-cloud-run.sh
```

## Step 8: Get Your App URL

After deployment, Cloud Run will give you a URL like:
```
https://aus-freight-dashboard-xxxxx-uc.a.run.app
```

## Step 9: Update Automation Script

Update `simple_automation.py`:
```python
self.streamlit_cloud_url = "https://aus-freight-dashboard-xxxxx-uc.a.run.app"
```

## Step 10: Test Your Deployment

1. Visit the Cloud Run URL in your browser
2. Test the dashboard functionality
3. Check that data loads correctly

## Monitoring & Management

### View Logs:
```bash
gcloud run services logs read aus-freight-dashboard --region us-central1
```

### Update Deployment:
Just run the deployment script again - it will update the existing service

### Delete Service (if needed):
```bash
gcloud run services delete aus-freight-dashboard --region us-central1
```

## Cost Estimation

- **Free Tier**: 2 million requests/month
- **After Free Tier**: ~$0.40 per million requests
- **Memory/CPU**: ~$0.00002400 per GB-second
- **Estimated Monthly Cost**: $5-15 for moderate usage

## Troubleshooting

### "Permission denied" errors:
```bash
gcloud auth login
gcloud auth configure-docker
```

### Docker build fails:
- Make sure Docker Desktop is running
- Check Dockerfile syntax

### Deployment fails:
- Check that all APIs are enabled
- Verify billing is enabled
- Check Cloud Run logs for errors

### App crashes:
- Increase memory: `--memory 8Gi` → `--memory 16Gi`
- Increase timeout: `--timeout 600` → `--timeout 900`
- Check logs for specific errors

## Next Steps

1. Set up automated deployment from GitHub (optional)
2. Configure custom domain (optional)
3. Set up monitoring alerts (optional)

