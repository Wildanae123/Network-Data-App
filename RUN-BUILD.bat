@echo off
setlocal enabledelayedexpansion
title Build and Run Standalone Network Data App

SET "FRONTEND_DIST_DIR=dist"
SET "PORT=8000"
SET "APP_EXE_PATH=dist\NetworkDataApp.exe"

echo ================================================================
echo.
echo   Building and Running Standalone Network Data App...
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

REM Check if Node.js is available (only needed for the initial build step if 'build_standalone.py' uses it)
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js is not installed or not in PATH. This might be required by build_standalone.py.
    echo If the build fails, please install Node.js.
    rem We'll let it proceed for now as it might not be strictly needed for *this* script's logic
)

REM Check if build script exists
if not exist "required\build_standalone.py" (
    echo ERROR: build_standalone.py not found!
    echo Please ensure the build script is in the project root directory.
    pause
    exit /b 1
)

REM --- Check if already built and prompt user ---
if exist "%FRONTEND_DIST_DIR%\index.html" (
    echo.
    echo A previous build was detected in '%FRONTEND_DIST_DIR%'.
    echo.
    echo Please choose an option:
    echo   1. Rebuild the application
    echo   2. Start the existing build
    echo.

    CHOICE /C 12 /M "Enter your choice (1 or 2)"

    if errorlevel 2 (
        goto :RUN_APP
    ) else if errorlevel 1 (
        goto :BUILD_APP
    ) else (
        echo An unexpected error occurred with your input. Exiting.
        pause
        exit /b 1
    )
)

:BUILD_APP
echo.
echo Building standalone executable and frontend...
python required\build_standalone.py

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Build Complete!
echo
echo   Your standalone executable (backend) is ready:
echo   !APP_EXE_PATH!
echo
echo   The frontend (web interface) is in:
echo   !FRONTEND_DIST_DIR!
echo ================================================================
echo.

:RUN_APP
echo.
echo Starting the Network Data App...
echo DEBUG_POINT_1: After 'Starting the Network Data App...'

REM --- Start the Python backend executable (if it exists and is needed) ---
echo DEBUG_POINT_2: Before 'if exist' check. APP_EXE_PATH is: "!APP_EXE_PATH!"
if exist "!APP_EXE_PATH!" (
    echo DEBUG_POINT_3: Inside IF block (backend executable found)
    echo Starting the backend application (!APP_EXE_PATH!)...
    start "" /B "!APP_EXE_PATH!"
    echo DEBUG_POINT_4: Backend start command issued. Waiting for initialization...
    echo Backend started. Waiting a moment for it to initialize...
    timeout /t 5 >nul
    echo DEBUG_POINT_5: Timeout finished.
) else (
    echo DEBUG_POINT_6: Inside ELSE block (backend executable NOT found).
    echo WARNING: Backend executable (!APP_EXE_PATH!) not found.
    echo If your frontend requires a backend, it might not function correctly.
    echo DEBUG_POINT_7: Warning displayed.
)
echo DEBUG_POINT_8: After IF/ELSE block for backend.

REM --- Start Python's Simple HTTP Server for the frontend ---
echo.
echo DEBUG_POINT_9: Before serving frontend. FRONTEND_DIST_DIR: "!FRONTEND_DIST_DIR!", PORT: "!PORT!"
echo Serving the frontend from "!FRONTEND_DIST_DIR!" on http://localhost:!PORT!...
echo To stop the server, close this console window.
echo.

REM Open the browser
start "" "http://localhost:!PORT!"
echo DEBUG_POINT_10: Browser open command issued.

REM Start the Python Simple HTTP Server.
REM We need to navigate to the dist directory first.
cd /d "!FRONTEND_DIST_DIR!"
echo DEBUG_POINT_11: Changed directory to "!CD%".
python -m http.server !PORT!
echo DEBUG_POINT_12: Python server started.

REM Go back to the original directory (optional, but good practice)
cd /d "%~dp0"
echo DEBUG_POINT_13: Changed directory back to original.

echo.
echo Server stopped.
pause