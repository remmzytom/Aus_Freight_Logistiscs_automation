# PowerShell script to start Streamlit app
Write-Host "Starting Australian Freight Export Analysis Dashboard..." -ForegroundColor Green
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Start Streamlit app (will auto-open browser)
streamlit run app.py --server.headless=false

