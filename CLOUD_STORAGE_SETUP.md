# Cloud Storage Setup Guide

This guide will help you set up Google Cloud Storage so your automation can upload data and your Cloud Run dashboard can automatically access the latest data.

## Prerequisites

- Google Cloud Project: `manifest-wind-478914-p8`
- Google Cloud SDK installed and configured
- Cloud Run service already deployed

## Step 1: Create Cloud Storage Bucket

Run this command in Google Cloud SDK Shell:

```bash
gsutil mb -p manifest-wind-478914-p8 -l us-central1 gs://aus-freight-logistics-data
```

Or use the Cloud Console:
1. Go to: https://console.cloud.google.com/storage
2. Click "Create Bucket"
3. Name: `aus-freight-logistics-data`
4. Location: `us-central1` (same as Cloud Run)
5. Click "Create"

## Step 2: Set Bucket Permissions

### Option 1: Make bucket publicly readable (Easiest)

Run this in Google Cloud SDK Shell:

```batch
gsutil iam ch allUsers:objectViewer gs://aus-freight-logistics-data
```

### Option 2: Allow Cloud Run service account (More Secure)

**For Windows Command Prompt:**

```batch
for /f "tokens=*" %i in ('gcloud projects describe manifest-wind-478914-p8 --format="value(projectNumber)"') do set PROJECT_NUMBER=%i
set SERVICE_ACCOUNT=%PROJECT_NUMBER%-compute@developer.gserviceaccount.com
gsutil iam ch serviceAccount:%SERVICE_ACCOUNT%:objectViewer gs://aus-freight-logistics-data
```

**For PowerShell:**

```powershell
$PROJECT_NUMBER = gcloud projects describe manifest-wind-478914-p8 --format="value(projectNumber)"
$SERVICE_ACCOUNT = "$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectViewer gs://aus-freight-logistics-data
```

**Or manually get project number:**

1. Go to: https://console.cloud.google.com/iam-admin/settings?project=manifest-wind-478914-p8
2. Copy the "Project number" (e.g., `828544570472`)
3. Run:
```batch
gsutil iam ch serviceAccount:828544570472-compute@developer.gserviceaccount.com:objectViewer gs://aus-freight-logistics-data
```
(Replace `828544570472` with your actual project number)

## Step 3: Configure Authentication

### For Automation (Local Machine)

1. Authenticate with Google Cloud:
```bash
gcloud auth application-default login
```

2. Set the project:
```bash
gcloud config set project manifest-wind-478914-p8
```

### For Cloud Run

Cloud Run automatically uses the service account. Make sure the service account has Storage Object Viewer permission (done in Step 2).

## Step 4: Test the Setup

### Test Upload (from automation script)

After running automation, check if files are uploaded:

```bash
gsutil ls gs://aus-freight-logistics-data/
```

You should see:
- `exports_cleaned.csv`
- `exports_2024_2025.csv` (optional)

### Test Download (from Cloud Run)

The dashboard will automatically download from Cloud Storage when it starts.

## Step 5: Update Configuration (if needed)

If you want to use a different bucket name, update these files:

1. **simple_automation.py** (line ~65):
   ```python
   self.gcs_bucket_name = "your-bucket-name"
   ```

2. **app.py** (line ~330):
   ```python
   gcs_bucket_name = "your-bucket-name"
   ```

## How It Works

1. **Automation runs** → Processes data → Saves to `data/exports_cleaned.csv`
2. **Automation uploads** → Uploads `exports_cleaned.csv` to Cloud Storage
3. **Cloud Run dashboard starts** → Downloads latest file from Cloud Storage
4. **Dashboard displays** → Shows fresh data automatically

## Troubleshooting

### "Bucket not found" error
- Make sure bucket name matches exactly: `aus-freight-logistics-data`
- Verify bucket exists: `gsutil ls gs://aus-freight-logistics-data/`

### "Permission denied" error
- Check IAM permissions (Step 2)
- Verify authentication: `gcloud auth list`

### "Module not found: google.cloud.storage"
- Install: `pip install google-cloud-storage`
- Or update requirements.txt and redeploy

### Dashboard still shows old data
- Check if file was uploaded: `gsutil ls -l gs://aus-freight-logistics-data/exports_cleaned.csv`
- Clear Cloud Run cache (restart the service)
- Check Cloud Run logs for download errors

## Benefits

✅ **Automatic Updates**: Dashboard always shows latest data  
✅ **No Redeployment**: Data updates without rebuilding Docker image  
✅ **Centralized Storage**: Single source of truth for data  
✅ **Scalable**: Works with multiple Cloud Run instances  

## Next Steps

1. Create the bucket (Step 1)
2. Set permissions (Step 2)
3. Run automation - it will upload automatically
4. Refresh Cloud Run dashboard - it will download automatically

Your dashboard will now update automatically whenever automation runs! 🎉

