# Google Cloud Run Deployment Script for Windows PowerShell
# This script builds and deploys your Streamlit app to Cloud Run

# Configuration - UPDATE THESE VALUES
$PROJECT_ID = "manifest-wind-478914-p8"  # Your GCP project ID
$SERVICE_NAME = "aus-freight-dashboard"
$REGION = "us-central1"  # Change to your preferred region
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "=== Google Cloud Run Deployment ===" -ForegroundColor Green
Write-Host ""

# Check if gcloud is installed
try {
    gcloud --version | Out-Null
} catch {
    Write-Host "Error: gcloud CLI is not installed" -ForegroundColor Red
    Write-Host "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running" -ForegroundColor Red
    Write-Host "Please start Docker Desktop"
    exit 1
}

# Set the project
Write-Host "Setting GCP project to $PROJECT_ID..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE_NAME .

# Push image to Google Container Registry
Write-Host "Pushing image to Google Container Registry..." -ForegroundColor Yellow
docker push $IMAGE_NAME

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..." -ForegroundColor Yellow
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

# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

Write-Host ""
Write-Host "=== Deployment Complete! ===" -ForegroundColor Green
Write-Host "Your app is available at: $SERVICE_URL" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Update simple_automation.py with the new URL: $SERVICE_URL"
Write-Host "2. Test the dashboard at the URL above"
Write-Host ""

