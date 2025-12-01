# Australian Freight Logistics Automation - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Components](#architecture--components)
3. [Complete Setup Process](#complete-setup-process)
4. [Data Pipeline Flow](#data-pipeline-flow)
5. [Deployment Architecture](#deployment-architecture)
6. [Automation Workflow](#automation-workflow)
7. [Technologies & Dependencies](#technologies--dependencies)
8. [Project Structure](#project-structure)
9. [Configuration & Secrets](#configuration--secrets)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Maintenance & Updates](#maintenance--updates)

---

## Project Overview

### Purpose
This project automates the collection, cleaning, analysis, and visualization of Australian freight export data from the Australian Bureau of Statistics (ABS). The system runs weekly to ensure stakeholders always have access to the latest freight logistics insights through an interactive web dashboard.

### Key Features
- ✅ **Automated Weekly Data Collection** - Runs every Monday at 9 AM Sydney time (AEDT)
- ✅ **Cloud-Based Infrastructure** - Fully deployed on Google Cloud Platform
- ✅ **Real-Time Dashboard** - Interactive Streamlit dashboard deployed on Cloud Run
- ✅ **Automatic Data Updates** - Dashboard refreshes automatically with new data
- ✅ **Email Notifications** - Success/failure notifications sent after each automation run
- ✅ **Scalable Architecture** - Handles 1.5M+ records efficiently
- ✅ **Zero Manual Intervention** - Fully automated end-to-end process

### Business Value
- **Time Savings**: Eliminates manual data collection and processing
- **Data Accuracy**: Automated pipeline reduces human error
- **Real-Time Insights**: Stakeholders access latest data immediately
- **Cost Efficiency**: Cloud-based infrastructure scales automatically
- **Reliability**: Automated scheduling ensures consistent updates

---

## Architecture & Components

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS (CI/CD)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Weekly Automation Workflow                              │  │
│  │  Schedule: Monday 9 AM Sydney (Sunday 10 PM UTC)        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              AUTOMATION PIPELINE (simple_automation.py)         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  1. Data     │→ │  2. Data     │→ │  3. Data     │         │
│  │  Collection  │  │  Cleaning    │  │  Analysis    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                │
│         └──────────────────┴──────────────────┘                │
│                            │                                    │
│                            ▼                                    │
│              ┌──────────────────────────┐                       │
│              │  4. Upload to Cloud       │                       │
│              │     Storage (GCS)         │                       │
│              └──────────────────────────┘                       │
└────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              GOOGLE CLOUD STORAGE (GCS)                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Bucket: aus-freight-logistics-data                       │ │
│  │  Files:                                                   │ │
│  │    - exports_cleaned.csv (1.5M+ records)                 │ │
│  │    - exports_2024_2025.csv (raw data)                    │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              GOOGLE CLOUD RUN (Dashboard)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Streamlit Dashboard (app.py)                          │  │
│  │  - Downloads latest data from GCS on startup            │  │
│  │  - Interactive visualizations                           │  │
│  │  - Real-time KPIs                                        │  │
│  │  - Filterable date ranges                                │  │
│  │  URL: aus-freight-dashboard-*.run.app                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **GitHub Actions Workflows**
- **Location**: `.github/workflows/`
- **Files**:
  - `weekly_automation.yml` - Weekly data automation pipeline
  - `deploy-cloud-run.yml` - Automatic Cloud Run deployment

#### 2. **Automation Script**
- **File**: `simple_automation.py`
- **Purpose**: Orchestrates the complete data pipeline
- **Functions**:
  - Data collection from ABS website
  - Data cleaning and processing
  - Data analysis and insights generation
  - Cloud Storage upload
  - Email notifications

#### 3. **Dashboard Application**
- **File**: `app.py`
- **Framework**: Streamlit
- **Features**:
  - Interactive data visualizations
  - KPI metrics display
  - Date range filtering
  - Top/Lowest port analysis
  - Country-wise export analysis

#### 4. **Data Storage**
- **Platform**: Google Cloud Storage
- **Bucket**: `aus-freight-logistics-data`
- **Files**: CSV format (1.5M+ records)

#### 5. **Containerization**
- **File**: `Dockerfile`
- **Base Image**: Python 3.11-slim
- **Deployment**: Google Cloud Run

---

## Complete Setup Process

### Prerequisites

1. **Google Cloud Platform Account**
   - Project ID: `manifest-wind-478914-p8`
   - Billing enabled
   - APIs enabled (Cloud Run, Cloud Storage, Container Registry)

2. **GitHub Account**
   - Repository: `Aus_Freight_Logistiscs_automation`
   - GitHub Actions enabled

3. **Local Development Environment**
   - Python 3.11+
   - Google Cloud SDK installed
   - Docker Desktop installed (for local testing)

### Step-by-Step Setup

#### Phase 1: Google Cloud Platform Setup

**1.1. Create Google Cloud Project**
```bash
# Set project
gcloud config set project manifest-wind-478914-p8

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage-component.googleapis.com
```

**1.2. Create Cloud Storage Bucket**
```bash
# Create bucket
gsutil mb -p manifest-wind-478914-p8 -l us-central1 gs://aus-freight-logistics-data

# Set permissions (allow Cloud Run to read)
PROJECT_NUMBER=$(gcloud projects describe manifest-wind-478914-p8 --format="value(projectNumber)")
gsutil iam ch serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com:objectViewer gs://aus-freight-logistics-data
```

**1.3. Create Service Account for GitHub Actions**
```bash
# Create service account
gcloud iam service-accounts create github-actions-sa \
    --display-name="GitHub Actions Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding manifest-wind-478914-p8 \
    --member="serviceAccount:github-actions-sa@manifest-wind-478914-p8.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding manifest-wind-478914-p8 \
    --member="serviceAccount:github-actions-sa@manifest-wind-478914-p8.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding manifest-wind-478914-p8 \
    --member="serviceAccount:github-actions-sa@manifest-wind-478914-p8.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create gcp-sa-key.json \
    --iam-account=github-actions-sa@manifest-wind-478914-p8.iam.gserviceaccount.com
```

#### Phase 2: GitHub Configuration

**2.1. Configure GitHub Secrets**

Go to: `Settings → Secrets and variables → Actions`

Add the following secrets:

| Secret Name | Description | Example Value |
|------------|-------------|---------------|
| `GCP_SA_KEY` | Service account key JSON (from Phase 1.3) | `{"type": "service_account", ...}` |
| `GCS_CREDENTIALS` | Cloud Storage credentials JSON | `{"type": "service_account", ...}` |
| `SMTP_SERVER` | Email SMTP server | `smtp.gmail.com` |
| `SENDER_EMAIL` | Email sender address | `your_email@gmail.com` |
| `SENDER_PASSWORD` | Email app password | `xxxx xxxx xxxx xxxx` |
| `RECIPIENT_EMAILS` | Comma-separated recipient emails | `email1@example.com,email2@example.com` |

**2.2. Verify Workflow Files**

Ensure these files exist:
- `.github/workflows/weekly_automation.yml`
- `.github/workflows/deploy-cloud-run.yml`

#### Phase 3: Local Development Setup

**3.1. Clone Repository**
```bash
git clone https://github.com/remmzytom/Aus_Freight_Logistiscs_automation.git
cd Aus_Freight_Logistiscs_automation
```

**3.2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3.3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3.4. Authenticate with Google Cloud**
```bash
gcloud auth application-default login
gcloud config set project manifest-wind-478914-p8
```

**3.5. Test Local Automation**
```bash
python simple_automation.py
```

**3.6. Test Local Dashboard**
```bash
streamlit run app.py
```

#### Phase 4: Initial Deployment

**4.1. Deploy to Cloud Run (Manual)**

Option A: Using GitHub Actions (Recommended)
- Push code to `main` branch
- GitHub Actions will automatically deploy

Option B: Using Batch Script (Windows)
```bash
.\deploy-cloud-run.bat
```

**4.2. Verify Deployment**
- Check Cloud Run service URL
- Access dashboard and verify data loads
- Check Cloud Storage bucket for uploaded files

**4.3. Test Automation**
- Manually trigger GitHub Actions workflow
- Verify data uploads to Cloud Storage
- Verify dashboard updates with new data

---

## Data Pipeline Flow

### Complete Data Journey

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA COLLECTION                                        │
│ ────────────────────────────────────────────────────────────── │
│ Script: 2024_2025_extractor.py                                │
│ Source: ABS Website (Australian Bureau of Statistics)         │
│ Output: data/exports_2024_2025.csv (Raw CSV file)             │
│ Size: ~500MB - 1GB (1.5M+ records)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA CLEANING                                          │
│ ────────────────────────────────────────────────────────────── │
│ Script: data_cleaning.ipynb (via simple_automation.py)        │
│ Process:                                                       │
│   - Remove duplicates                                          │
│   - Standardize country names                                 │
│   - Clean port names                                           │
│   - Format dates                                               │
│   - Handle missing values                                      │
│   - Data type conversions                                      │
│ Output: data/exports_cleaned.csv                               │
│ Size: ~400-800MB (1.5M+ cleaned records)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: DATA ANALYSIS                                          │
│ ────────────────────────────────────────────────────────────── │
│ Script: data_analysis.ipynb (via simple_automation.py)       │
│ Process:                                                       │
│   - Calculate KPIs (Total Value, Total Weight, etc.)          │
│   - Generate insights                                         │
│   - Create summary statistics                                 │
│   - Identify trends                                           │
│ Output: Analysis results (logged)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: CLOUD STORAGE UPLOAD                                   │
│ ────────────────────────────────────────────────────────────── │
│ Script: simple_automation.py → upload_to_cloud_storage()      │
│ Destination: gs://aus-freight-logistics-data/                │
│ Files Uploaded:                                                │
│   - exports_cleaned.csv (Primary file for dashboard)          │
│   - exports_2024_2025.csv (Backup raw data)                  │
│ Authentication: Service Account via GCS_CREDENTIALS secret    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: DASHBOARD DATA LOAD                                    │
│ ────────────────────────────────────────────────────────────── │
│ Script: app.py → ensure_data_file()                            │
│ Process:                                                       │
│   1. Check Cloud Storage for latest exports_cleaned.csv      │
│   2. Download if available (or use local fallback)            │
│   3. Load data in chunks (memory-efficient)                   │
│   4. Process and cache for dashboard                          │
│ Result: Dashboard displays latest data automatically           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Processing Details

#### Data Collection (`2024_2025_extractor.py`)
- **Method**: Web scraping from ABS website
- **Frequency**: Weekly (every Monday)
- **Data Range**: 2024-2025 export records
- **Output Format**: CSV
- **Error Handling**: Retry logic, logging

#### Data Cleaning (`data_cleaning.ipynb`)
- **Chunk Processing**: Handles large files efficiently
- **Cleaning Steps**:
  1. Remove duplicate records
  2. Standardize country names (using `country_mapping.py`)
  3. Clean port names
  4. Format date columns
  5. Handle encoding issues
  6. Convert data types (numeric, dates)
  7. Remove invalid records

#### Data Analysis (`data_analysis.ipynb`)
- **KPI Calculations**:
  - Total Export Value
  - Total Export Weight
  - Average Value per Transaction
  - Top Export Destinations
  - Top Export Ports
  - Monthly Trends

---

## Deployment Architecture

### Cloud Run Deployment

**Configuration:**
- **Service Name**: `aus-freight-dashboard`
- **Region**: `us-central1`
- **Memory**: 8GB
- **CPU**: 4 vCPU
- **Timeout**: 600 seconds (10 minutes)
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Port**: 8080
- **Authentication**: Public (unauthenticated)

**Docker Image:**
- **Base**: `python:3.11-slim`
- **Registry**: Google Container Registry (GCR)
- **Image**: `gcr.io/manifest-wind-478914-p8/aus-freight-dashboard`

**Deployment Process:**
1. Code pushed to `main` branch
2. GitHub Actions workflow triggered
3. Docker image built
4. Image pushed to GCR
5. Cloud Run service updated
6. New version deployed

### Cloud Storage Integration

**Bucket Configuration:**
- **Name**: `aus-freight-logistics-data`
- **Location**: `us-central1`
- **Storage Class**: Standard
- **Access**: Service account-based

**File Structure:**
```
gs://aus-freight-logistics-data/
├── exports_cleaned.csv          (Primary dashboard data)
└── exports_2024_2025.csv        (Raw backup data)
```

**Permissions:**
- Cloud Run service account: `objectViewer` (read access)
- GitHub Actions service account: `storage.admin` (write access)

---

## Automation Workflow

### Weekly Automation Schedule

**Schedule**: Every Monday at 9:00 AM Sydney time (AEDT)
- **UTC Equivalent**: Sunday 10:00 PM UTC
- **Cron Expression**: `"0 22 * * 0"`

**Workflow File**: `.github/workflows/weekly_automation.yml`

### Automation Steps

#### Step 1: Environment Setup
```yaml
- Checkout code from repository
- Set up Python 3.11
- Install dependencies from requirements.txt
- Configure Google Cloud credentials
- Create data directory
```

#### Step 2: Run Automation Pipeline
```python
python simple_automation.py
```

This executes:
1. **Data Collection** (`collect_data()`)
   - Downloads fresh data from ABS website
   - Saves to `data/exports_2024_2025.csv`

2. **Data Cleaning** (`run_data_cleaning()`)
   - Executes `data_cleaning.ipynb`
   - Processes raw data
   - Saves to `data/exports_cleaned.csv`

3. **Data Analysis** (`run_data_analysis()`)
   - Executes `data_analysis.ipynb`
   - Generates insights and KPIs

4. **Cloud Storage Upload** (`upload_to_cloud_storage()`)
   - Uploads `exports_cleaned.csv` to GCS
   - Uploads `exports_2024_2025.csv` (backup)

5. **Email Notification** (`send_success_notification()`)
   - Sends success email with summary
   - Includes automation duration and statistics

#### Step 3: Artifact Upload
- Upload logs (on failure)
- Upload results (CSV files)
- Upload executed notebooks
- Retention: 7 days

### Manual Trigger

The workflow can be manually triggered:
1. Go to GitHub repository
2. Click "Actions" tab
3. Select "Weekly Freight Data Automation"
4. Click "Run workflow"
5. Select branch and click "Run workflow"

### Error Handling

**On Failure:**
- Error logs uploaded as artifacts
- Failure email notification sent
- Workflow marked as failed
- Previous data remains available in Cloud Storage

**On Success:**
- Success email notification sent
- Data uploaded to Cloud Storage
- Dashboard automatically updates on next access

---

## Technologies & Dependencies

### Core Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11 | Programming language |
| Streamlit | ≥1.31.0 | Dashboard framework |
| Pandas | ≥2.1.0 | Data processing |
| Plotly | ≥5.17.0 | Interactive visualizations |
| Google Cloud Run | Latest | Container hosting |
| Google Cloud Storage | Latest | Data storage |
| GitHub Actions | Latest | CI/CD automation |
| Docker | Latest | Containerization |

### Python Dependencies

**Core Libraries:**
```python
streamlit>=1.31.0          # Dashboard framework
pandas>=2.1.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
plotly>=5.17.0             # Interactive charts
matplotlib>=3.7.0          # Static plots
seaborn>=0.12.0            # Statistical visualization
```

**Cloud & Infrastructure:**
```python
google-cloud-storage>=2.10.0  # Cloud Storage integration
```

**Data Processing:**
```python
openpyxl>=3.1.0            # Excel file handling
xlrd>=2.0.0                # Excel reading
requests>=2.31.0           # HTTP requests
```

**Utilities:**
```python
jupyter>=1.0.0             # Notebook execution
nbconvert>=7.0.0           # Notebook conversion
scikit-learn>=1.3.0        # Machine learning utilities
Pillow>=10.0.0             # Image processing
schedule>=1.2.0            # Local scheduling (optional)
```

### Infrastructure Components

**Google Cloud Platform:**
- Cloud Run (Serverless container hosting)
- Cloud Storage (Object storage)
- Container Registry (Docker image storage)
- IAM (Identity and access management)

**GitHub:**
- GitHub Actions (CI/CD)
- GitHub Secrets (Secure configuration)
- GitHub Repository (Code hosting)

---

## Project Structure

```
Aus_Freight_Logistic/
│
├── .github/
│   └── workflows/
│       ├── weekly_automation.yml      # Weekly automation workflow
│       └── deploy-cloud-run.yml      # Auto-deployment workflow
│
├── data/                              # Data files (not in Git)
│   ├── exports_2024_2025.csv         # Raw data from ABS
│   └── exports_cleaned.csv            # Processed data
│
├── pipeline/
│   ├── __init__.py
│   └── data_pipeline.py              # Data pipeline utilities
│
├── app.py                             # Streamlit dashboard (main)
├── simple_automation.py              # Automation orchestrator
├── 2024_2025_extractor.py            # Data collection script
├── data_cleaning.ipynb               # Data cleaning notebook
├── data_analysis.ipynb               # Data analysis notebook
├── scheduler.py                      # Local scheduler (optional)
│
├── Dockerfile                         # Docker container definition
├── requirements.txt                   # Python dependencies
├── .dockerignore                     # Docker ignore patterns
│
├── deploy-cloud-run.bat              # Windows deployment script
├── setup-cloud-storage.bat           # Cloud Storage setup script
├── test_cloud_storage.py             # Cloud Storage test script
│
├── country_mapping.py                # Country name standardization
├── region_mapping.py                 # Region mapping utilities
├── sitc_mapping.py                   # SITC code mapping
│
├── README.md                         # Project overview
├── COMPLETE_PROJECT_DOCUMENTATION.md # This file
├── CLOUD_RUN_SETUP.md                # Cloud Run setup guide
├── CLOUD_STORAGE_SETUP.md            # Cloud Storage setup guide
├── GITHUB_ACTIONS_DEPLOYMENT_SETUP.md # GitHub Actions guide
├── QUICK_START_CLOUD_RUN.md          # Quick start guide
└── Project_Overview_Stakeholder_Report.md # Stakeholder report
```

### Key Files Explained

**`app.py`** (2,746 lines)
- Main Streamlit dashboard application
- Handles data loading from Cloud Storage
- Implements all visualizations and KPIs
- Manages date filtering and user interactions

**`simple_automation.py`** (422 lines)
- Main automation orchestrator
- Coordinates data collection, cleaning, analysis
- Handles Cloud Storage uploads
- Manages email notifications

**`2024_2025_extractor.py`**
- Web scraper for ABS website
- Downloads export data
- Handles retries and error cases

**`data_cleaning.ipynb`**
- Jupyter notebook for data cleaning
- Executed programmatically by automation
- Processes raw CSV into cleaned format

**`data_analysis.ipynb`**
- Jupyter notebook for data analysis
- Generates insights and KPIs
- Creates summary statistics

**`Dockerfile`**
- Container definition for Cloud Run
- Based on Python 3.11-slim
- Installs dependencies and runs Streamlit

---

## Configuration & Secrets

### GitHub Secrets Configuration

**Required Secrets:**

1. **`GCP_SA_KEY`**
   - **Type**: JSON
   - **Content**: Service account key for Cloud Run deployment
   - **Permissions**: `roles/run.admin`, `roles/iam.serviceAccountUser`
   - **Usage**: GitHub Actions deployment workflow

2. **`GCS_CREDENTIALS`**
   - **Type**: JSON
   - **Content**: Service account key for Cloud Storage access
   - **Permissions**: `roles/storage.admin`
   - **Usage**: Automation workflow for uploading data

3. **`SMTP_SERVER`**
   - **Type**: String
   - **Example**: `smtp.gmail.com`
   - **Usage**: Email notifications

4. **`SENDER_EMAIL`**
   - **Type**: String
   - **Example**: `your_email@gmail.com`
   - **Usage**: Email sender address

5. **`SENDER_PASSWORD`**
   - **Type**: String
   - **Note**: Use app-specific password for Gmail
   - **Usage**: Email authentication

6. **`RECIPIENT_EMAILS`**
   - **Type**: String
   - **Format**: Comma-separated emails
   - **Example**: `email1@example.com,email2@example.com`
   - **Usage**: Email recipients

### Environment Variables

**In `simple_automation.py`:**
```python
# Email configuration (can use env vars)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'default@example.com')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')
RECIPIENT_EMAILS = os.getenv('RECIPIENT_EMAILS', '').split(',')

# Cloud Storage configuration
GCS_BUCKET_NAME = "aus-freight-logistics-data"
GCS_CLEANED_FILE = "exports_cleaned.csv"
GCS_RAW_FILE = "exports_2024_2025.csv"
```

**In `app.py`:**
```python
# Cloud Storage configuration
GCS_BUCKET_NAME = "aus-freight-logistics-data"
GCS_CLEANED_FILE = "exports_cleaned.csv"
```

### Cloud Run Configuration

**Environment Variables** (if needed):
- None currently required (uses default service account)

**Service Account:**
- Uses default compute service account
- Requires `Storage Object Viewer` permission on Cloud Storage bucket

---

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. Automation Not Running

**Symptoms:**
- GitHub Actions workflow not triggering
- No data updates in dashboard

**Solutions:**
- Check workflow schedule: `"0 22 * * 0"` (Sunday 10 PM UTC)
- Verify workflow file exists: `.github/workflows/weekly_automation.yml`
- Check GitHub Actions is enabled in repository settings
- Manually trigger workflow to test

#### 2. Cloud Storage Upload Fails

**Symptoms:**
- Error: "Permission denied"
- Error: "Bucket not found"

**Solutions:**
```bash
# Verify bucket exists
gsutil ls gs://aus-freight-logistics-data/

# Check permissions
gsutil iam get gs://aus-freight-logistics-data/

# Verify credentials
gcloud auth application-default print-access-token

# Check GitHub secret GCS_CREDENTIALS is set correctly
```

#### 3. Dashboard Not Loading Data

**Symptoms:**
- Dashboard shows "No data available"
- Error loading CSV file

**Solutions:**
- Check Cloud Storage bucket has `exports_cleaned.csv`
- Verify Cloud Run service account has `objectViewer` permission
- Check Cloud Run logs: `gcloud run services logs read aus-freight-dashboard --region us-central1`
- Verify file exists: `gsutil ls -l gs://aus-freight-logistics-data/exports_cleaned.csv`

#### 4. Docker Build Fails

**Symptoms:**
- GitHub Actions deployment fails
- Error building Docker image

**Solutions:**
- Check `Dockerfile` syntax
- Verify `requirements.txt` has all dependencies
- Test locally: `docker build -t test-image .`
- Check `.dockerignore` excludes unnecessary files

#### 5. Email Notifications Not Sending

**Symptoms:**
- No email received after automation
- SMTP authentication error

**Solutions:**
- Verify GitHub secrets are set correctly
- For Gmail, use app-specific password (not regular password)
- Check SMTP server and port settings
- Test email configuration locally

#### 6. Memory Issues with Large Data

**Symptoms:**
- Dashboard crashes
- "Out of memory" errors

**Solutions:**
- Increase Cloud Run memory: `--memory 8Gi` (already configured)
- Verify chunked data loading in `app.py`
- Check data file size: `gsutil du -sh gs://aus-freight-logistics-data/exports_cleaned.csv`

#### 7. Timezone Issues

**Symptoms:**
- Automation runs at wrong time
- Date filters show incorrect ranges

**Solutions:**
- GitHub Actions uses UTC
- Sydney time (AEDT) = UTC+11, (AEST) = UTC+10
- Current schedule: `"0 22 * * 0"` = Sunday 10 PM UTC = Monday 9 AM AEDT
- Adjust cron if needed for daylight saving changes

### Debugging Commands

**Check Cloud Run Status:**
```bash
gcloud run services describe aus-freight-dashboard --region us-central1
```

**View Cloud Run Logs:**
```bash
gcloud run services logs read aus-freight-dashboard --region us-central1 --limit 50
```

**Check Cloud Storage Files:**
```bash
gsutil ls -lh gs://aus-freight-logistics-data/
```

**Test Cloud Storage Access:**
```bash
python test_cloud_storage.py
```

**View GitHub Actions Logs:**
- Go to repository → Actions tab
- Click on workflow run
- View detailed logs

**Test Local Automation:**
```bash
python simple_automation.py
```

**Test Local Dashboard:**
```bash
streamlit run app.py
```

---

## Maintenance & Updates

### Regular Maintenance Tasks

#### Weekly
- ✅ **Automated**: Data collection and processing
- ✅ **Automated**: Dashboard data updates
- ⚠️ **Manual**: Review automation logs in GitHub Actions
- ⚠️ **Manual**: Verify dashboard is accessible

#### Monthly
- ⚠️ **Manual**: Review Cloud Storage costs
- ⚠️ **Manual**: Check Cloud Run usage and costs
- ⚠️ **Manual**: Review error logs for patterns
- ⚠️ **Manual**: Update dependencies if security patches available

#### Quarterly
- ⚠️ **Manual**: Review and update Python dependencies
- ⚠️ **Manual**: Review Cloud Run resource allocation
- ⚠️ **Manual**: Optimize data processing if needed
- ⚠️ **Manual**: Review and update documentation

### Updating Dependencies

**Process:**
1. Update `requirements.txt` with new versions
2. Test locally: `pip install -r requirements.txt`
3. Test automation: `python simple_automation.py`
4. Test dashboard: `streamlit run app.py`
5. Commit and push changes
6. GitHub Actions will rebuild and redeploy automatically

**Example:**
```bash
# Update a package
pip install --upgrade pandas

# Update requirements.txt
pip freeze > requirements.txt

# Test locally
python simple_automation.py
streamlit run app.py

# Commit and push
git add requirements.txt
git commit -m "Update pandas to latest version"
git push origin main
```

### Updating Dashboard Features

**Process:**
1. Make changes to `app.py`
2. Test locally: `streamlit run app.py`
3. Commit and push to `main` branch
4. GitHub Actions automatically deploys to Cloud Run
5. Verify deployment: Check Cloud Run URL

### Updating Automation Logic

**Process:**
1. Make changes to `simple_automation.py` or notebooks
2. Test locally: `python simple_automation.py`
3. Commit and push to `main` branch
4. Test via manual GitHub Actions trigger
5. Verify next scheduled run works correctly

### Scaling Considerations

**Current Configuration:**
- Cloud Run: 0-10 instances (auto-scaling)
- Memory: 8GB per instance
- CPU: 4 vCPU per instance
- Timeout: 600 seconds

**If Data Grows:**
- Increase Cloud Run memory if needed
- Consider increasing max instances
- Optimize data loading in `app.py`
- Consider data partitioning strategies

**Cost Optimization:**
- Current: Pay-per-use (scales to zero)
- Monitor Cloud Storage storage costs
- Review Cloud Run execution time
- Consider data retention policies

---

## Project Timeline & Evolution

### Initial Phase
- ✅ Basic data collection script
- ✅ Local data processing
- ✅ Simple Streamlit dashboard
- ✅ Manual execution

### Cloud Integration Phase
- ✅ Google Cloud Storage integration
- ✅ Cloud Run deployment
- ✅ Automated data uploads
- ✅ Dashboard auto-updates

### Automation Phase
- ✅ GitHub Actions workflow
- ✅ Weekly scheduled automation
- ✅ Email notifications
- ✅ Error handling and logging

### Optimization Phase
- ✅ Chunked data processing (handles 1.5M+ records)
- ✅ Memory-efficient loading
- ✅ Cloud Storage fallback
- ✅ Improved error handling

### Current State
- ✅ Fully automated weekly pipeline
- ✅ Cloud-based infrastructure
- ✅ Zero manual intervention required
- ✅ Scalable architecture
- ✅ Comprehensive error handling

---

## Future Enhancements

### Potential Improvements

1. **Data Analytics**
   - Machine learning predictions
   - Trend forecasting
   - Anomaly detection

2. **Dashboard Features**
   - Export reports (PDF/Excel)
   - Custom date range comparisons
   - Advanced filtering options
   - Real-time data refresh

3. **Infrastructure**
   - Multi-region deployment
   - CDN for faster loading
   - Database integration (BigQuery)
   - Caching layer

4. **Monitoring**
   - Cloud Monitoring dashboards
   - Alerting for failures
   - Performance metrics
   - Cost tracking

5. **Security**
   - Authentication for dashboard
   - API rate limiting
   - Data encryption at rest
   - Audit logging

---

## Support & Contact

### Documentation Files
- `README.md` - Quick start guide
- `COMPLETE_PROJECT_DOCUMENTATION.md` - This comprehensive guide
- `CLOUD_RUN_SETUP.md` - Cloud Run deployment guide
- `CLOUD_STORAGE_SETUP.md` - Cloud Storage setup guide
- `GITHUB_ACTIONS_DEPLOYMENT_SETUP.md` - GitHub Actions guide
- `QUICK_START_CLOUD_RUN.md` - Quick deployment guide

### Key URLs
- **Dashboard**: https://aus-freight-dashboard-828544570472.us-central1.run.app
- **GitHub Repository**: https://github.com/remmzytom/Aus_Freight_Logistiscs_automation
- **Cloud Console**: https://console.cloud.google.com/
- **GitHub Actions**: https://github.com/remmzytom/Aus_Freight_Logistiscs_automation/actions

### Getting Help
1. Check this documentation first
2. Review GitHub Actions logs
3. Check Cloud Run logs
4. Review error messages in automation.log
5. Test components individually

---

## Conclusion

This project represents a complete end-to-end automation solution for Australian freight logistics data. From data collection to visualization, every step is automated and runs reliably every week without manual intervention.

The architecture is scalable, cost-effective, and maintainable, leveraging Google Cloud Platform's serverless infrastructure and GitHub Actions for CI/CD automation.

**Key Achievements:**
- ✅ Fully automated weekly data pipeline
- ✅ Cloud-based scalable infrastructure
- ✅ Interactive real-time dashboard
- ✅ Zero manual intervention required
- ✅ Comprehensive error handling and notifications
- ✅ Handles 1.5M+ records efficiently

**Last Updated**: December 2024
**Project Status**: Production ✅
**Maintenance**: Automated weekly runs

---

*This documentation covers the complete project from initial setup to ongoing maintenance. For specific questions or issues, refer to the troubleshooting section or check the individual setup guides.*

