@echo off
title PaddyGuard AI - Streamlit Crop Disease Triage Platform
echo ========================================================
echo   PADDYGUARD AI - STREAMLIT TRIAGE AND DETECTION APP
echo   Languages: English, Telugu, Kannada
echo   Offline CIBRC Formulations and Voice Verification
echo ========================================================
echo.
cd /d "%~dp0"
if not exist "app.py" (
    if exist "paddyguard_vercel\app.py" (
        cd /d "%~dp0paddyguard_vercel"
    )
)
echo Starting Streamlit server with live reloading...
python -m streamlit run app.py --server.runOnSave true --server.port 8501
pause
