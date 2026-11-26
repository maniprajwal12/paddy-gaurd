@echo off
title Push PaddyGuard to GitHub (maniprajwal12/paddy-gaurd)
echo ========================================================
echo   UPDATING GITHUB: maniprajwal12/paddy-gaurd
echo ========================================================
echo.
cd /d "C:\Users\likit\OneDrive\Documents\Desktop\paddyguard_vercel"
git remote set-url origin https://github.com/maniprajwal12/paddy-gaurd.git
git branch -M main
git add -A
git diff --cached --quiet
if %ERRORLEVEL% neq 0 (
    echo Committing latest updates...
    git commit -m "Auto-sync latest PaddyGuard updates"
)
echo Pushing to GitHub...
git push -u origin main
echo.
echo ========================================================
echo   GITHUB UPDATE COMPLETE!
echo ========================================================
pause
