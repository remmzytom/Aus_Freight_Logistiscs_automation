@echo off
REM Google Cloud Run Deployment Script for Windows
REM This script builds and deploys your Streamlit app to Cloud Run

REM Configuration
set PROJECT_ID=manifest-wind-478914-p8
set SERVICE_NAME=aus-freight-dashboard
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo === Google Cloud Run Deployment ===
echo.

REM Check if gcloud is available
where gcloud >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: gcloud CLI is not in PATH
    echo Please add it to PATH or use the full path
    echo Default location: %LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker is not running
    echo Please start Docker Desktop
    pause
    exit /b 1
)

REM Set the project
echo Setting GCP project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Build the Docker image
echo Building Docker image...
docker build -t %IMAGE_NAME% .

REM Configure Docker to use gcloud as credential helper
echo Configuring Docker authentication...
gcloud auth configure-docker

REM Push image to Google Container Registry
echo Pushing image to Google Container Registry...
docker push %IMAGE_NAME%

REM Deploy to Cloud Run
echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_NAME% ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 8Gi ^
    --cpu 4 ^
    --timeout 600 ^
    --min-instances 0 ^
    --max-instances 10 ^
    --port 8080

REM Get the service URL
echo.
echo Getting service URL...
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo.
echo === Deployment Complete! ===
echo Your app is available at: %SERVICE_URL%
echo.
echo Next steps:
echo 1. Update simple_automation.py with the new URL: %SERVICE_URL%
echo 2. Test the dashboard at the URL above
echo.
pause

