@echo off
echo Starting Australian Freight Export Analysis Dashboard...
echo.
cd /d "%~dp0"
streamlit run app.py --server.headless=false
pause

