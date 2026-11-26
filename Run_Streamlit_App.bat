@echo off
title PaddyGuard AI - Streamlit Crop Disease Triage Platform
echo ========================================================
echo   PADDYGUARD AI - STREAMLIT TRIAGE AND DETECTION APP
echo   Languages: English, Telugu, Kannada
echo   Google Gemini 3.5 Multimodal AI and Offline CIBRC
echo ========================================================
echo.

set "TARGET_APP="
set "TARGET_DIR="

if exist "C:\Users\likit\OneDrive\Documents\Desktop\paddyguard_vercel\app.py" (
    set "TARGET_DIR=C:\Users\likit\OneDrive\Documents\Desktop\paddyguard_vercel"
    set "TARGET_APP=C:\Users\likit\OneDrive\Documents\Desktop\paddyguard_vercel\app.py"
) else if exist "C:\Users\likit\OneDrive\Documents\Desktop\app.py" (
    set "TARGET_DIR=C:\Users\likit\OneDrive\Documents\Desktop"
    set "TARGET_APP=C:\Users\likit\OneDrive\Documents\Desktop\app.py"
) else if exist "%~dp0app.py" (
    set "TARGET_DIR=%~dp0"
    set "TARGET_APP=%~dp0app.py"
) else if exist "%~dp0paddyguard_vercel\app.py" (
    set "TARGET_DIR=%~dp0paddyguard_vercel"
    set "TARGET_APP=%~dp0paddyguard_vercel\app.py"
)

if "%TARGET_APP%"=="" (
    echo [ERROR] Could not find app.py in any standard location.
    echo Please ensure app.py exists on your Desktop or in paddyguard_vercel.
    echo.
    pause
    exit /b 1
)

cd /d "%TARGET_DIR%"
echo Found Application: %TARGET_APP%
echo Working Directory: %CD%
echo.
echo Starting Streamlit server with live reloading...
python -m streamlit run "%TARGET_APP%" --server.runOnSave true --server.port 8501
if %ERRORLEVEL% neq 0 (
    echo.
    echo Streamlit stopped with exit code %ERRORLEVEL%.
    pause
)
