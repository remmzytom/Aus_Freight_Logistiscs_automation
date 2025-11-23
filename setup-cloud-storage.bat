@echo off
REM Cloud Storage Setup Script for Windows
REM This script sets up the Cloud Storage bucket and permissions

set PROJECT_ID=manifest-wind-478914-p8
set BUCKET_NAME=aus-freight-logistics-data
set REGION=us-central1

echo === Cloud Storage Setup ===
echo.

REM Step 1: Create bucket
echo Step 1: Creating Cloud Storage bucket...
gsutil mb -p %PROJECT_ID% -l %REGION% gs://%BUCKET_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo Warning: Bucket might already exist. Continuing...
)

echo.
echo Step 2: Setting bucket permissions...

REM Get project number
for /f "tokens=*" %%i in ('gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"') do set PROJECT_NUMBER=%%i

if "%PROJECT_NUMBER%"=="" (
    echo Error: Could not get project number
    echo Please set it manually:
    echo 1. Go to: https://console.cloud.google.com/iam-admin/settings?project=%PROJECT_ID%
    echo 2. Copy the Project number
    echo 3. Run: gsutil iam ch serviceAccount:[PROJECT_NUMBER]-compute@developer.gserviceaccount.com:objectViewer gs://%BUCKET_NAME%
    pause
    exit /b 1
)

set SERVICE_ACCOUNT=%PROJECT_NUMBER%-compute@developer.gserviceaccount.com
echo Project Number: %PROJECT_NUMBER%
echo Service Account: %SERVICE_ACCOUNT%

REM Grant Cloud Run service account access
gsutil iam ch serviceAccount:%SERVICE_ACCOUNT%:objectViewer gs://%BUCKET_NAME%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Setup Complete! ===
    echo Bucket: gs://%BUCKET_NAME%
    echo Service Account: %SERVICE_ACCOUNT%
    echo.
    echo Next steps:
    echo 1. Authenticate for automation: gcloud auth application-default login
    echo 2. Run automation - it will upload data automatically
    echo 3. Redeploy Cloud Run to include new dependencies
    echo.
) else (
    echo.
    echo Error setting permissions. Trying public access instead...
    gsutil iam ch allUsers:objectViewer gs://%BUCKET_NAME%
)

pause

