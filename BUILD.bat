@echo off
title Build Standalone Network Data App

echo ================================================================
echo.
echo   Building Standalone Network Data App...
echo.
echo ================================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH!
    echo Please install Node.js and try again.
    pause
    exit /b 1
)

echo Building standalone executable...
python build_standalone.py

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  Build Complete!
echo  
echo  Your standalone executable is ready:
echo  dist\NetworkDataApp.exe
echo  
echo  You can distribute this single file!
echo ================================================================
pause