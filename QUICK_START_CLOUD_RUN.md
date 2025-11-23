# Quick Start: Deploy to Google Cloud Run

## Prerequisites Checklist

- [ ] Google account (Gmail works)
- [ ] Credit card (for billing - $300 free credit)
- [ ] Docker Desktop installed and running
- [ ] Terminal/Command Prompt access

## Step-by-Step Deployment (15-20 minutes)

### Step 1: Create Google Cloud Account (5 min)

1. Go to: https://console.cloud.google.com/
2. Sign in with Google account
3. Click "Create Project"
   - Name: `aus-freight-dashboard`
   - Note your **Project ID** (looks like: `aus-freight-dashboard-123456`)
4. Enable billing (link credit card - $300 free credit)

### Step 2: Install Google Cloud SDK (5 min)

**Windows:**
- Download: https://cloud.google.com/sdk/docs/install
- Run installer, restart terminal

**Mac:**
```bash
brew install google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Step 3: Initialize gcloud (2 min)

Open terminal and run:
```bash
gcloud init
```

Follow prompts:
- Login with Google account
- Select your project
- Choose region: `us-central1` (or your preference)

### Step 4: Install Docker Desktop (if not installed)

- Download: https://www.docker.com/products/docker-desktop/
- Install and start Docker Desktop
- Verify: Docker icon in system tray

### Step 5: Configure Deployment Script (1 min)

**For Windows (PowerShell):**
1. Open `deploy-cloud-run.ps1`
2. Change line 5: `$PROJECT_ID = "your-project-id"` → Your actual Project ID
3. Save file

**For Mac/Linux:**
1. Open `deploy-cloud-run.sh`
2. Change line 5: `PROJECT_ID="your-project-id"` → Your actual Project ID
3. Save file

### Step 6: Deploy! (5-10 min)

**Windows (PowerShell):**
```powershell
cd "C:\Cursor AI_projects\Aus_Freight_Logistic"
.\deploy-cloud-run.ps1
```

**Mac/Linux:**
```bash
cd /path/to/Aus_Freight_Logistic
chmod +x deploy-cloud-run.sh
./deploy-cloud-run.sh
```

### Step 7: Get Your URL

After deployment completes, you'll see:
```
Your app is available at: https://aus-freight-dashboard-xxxxx-uc.a.run.app
```

### Step 8: Update Automation Script

Open `simple_automation.py` and update line 61:
```python
self.streamlit_cloud_url = "https://aus-freight-dashboard-xxxxx-uc.a.run.app"
```
(Replace with your actual Cloud Run URL)

## Manual Deployment Commands (Alternative)

If scripts don't work, run these commands manually:

```bash
# Set your project ID
PROJECT_ID="your-project-id"
SERVICE_NAME="aus-freight-dashboard"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Set project
gcloud config set project ${PROJECT_ID}

# Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build Docker image
docker build -t ${IMAGE_NAME} .

# Push to registry
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --timeout 600 \
    --min-instances 0 \
    --max-instances 10 \
    --port 8080

# Get URL
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
```

## Troubleshooting

### "gcloud: command not found"
- Install Google Cloud SDK (Step 2)
- Restart terminal

### "Docker is not running"
- Start Docker Desktop
- Wait for it to fully start (green icon)

### "Permission denied" during docker push
```bash
gcloud auth configure-docker
```

### Build fails
- Check Dockerfile syntax
- Make sure you're in the project directory
- Check Docker Desktop is running

### Deployment fails
- Verify billing is enabled
- Check all APIs are enabled
- Check Cloud Run logs: `gcloud run services logs read aus-freight-dashboard --region us-central1`

## Cost Information

- **Free Tier**: 2 million requests/month
- **After Free Tier**: ~$0.40 per million requests  
- **Memory/CPU**: ~$0.00002400 per GB-second
- **Estimated**: $5-15/month for moderate usage
- **$300 Free Credit**: Lasts months for this use case

## Next Steps After Deployment

1. ✅ Test your dashboard at the Cloud Run URL
2. ✅ Update `simple_automation.py` with the new URL
3. ✅ Test the automation script
4. ✅ Monitor usage in Google Cloud Console

## Updating Your App

To update after making changes:
1. Make your code changes
2. Run the deployment script again
3. Cloud Run will update automatically

## Need Help?

Check the detailed guide: `CLOUD_RUN_SETUP.md`

