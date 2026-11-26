@echo off
title Push PaddyGuard to GitHub (maniprajwal12/paddy-gaurd)
echo ========================================================
echo   UPDATING GITHUB: maniprajwal12/paddy-gaurd
echo ========================================================
echo.
cd /d "C:\Users\likit\OneDrive\Documents\Desktop\paddyguard_vercel"
git remote set-url origin https://github.com/maniprajwal12/paddy-gaurd.git
git branch -M main
echo Pushing to GitHub...
git push -u origin main
echo.
pause
