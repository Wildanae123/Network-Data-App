@echo off
title Network Data App - Standalone

if exist "dist\NetworkDataApp.exe" (
    echo Starting Network Data App...
    echo.
    echo The app will open in your default browser automatically.
    echo To stop the app, close this console window.
    echo.
    start dist\NetworkDataApp.exe
) else (
    echo NetworkDataApp.exe not found!
    echo Please run BUILD.bat first to create the standalone executable.
    pause
)