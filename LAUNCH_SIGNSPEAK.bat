@echo off
title SignSpeak Studio Launcher
cd /d "%~dp0"
echo ====================================================
echo Starting SignSpeak Studio...
echo ====================================================

if exist "dist\SignSpeak_Studio\SignSpeak_Studio.exe" (
    start "" "dist\SignSpeak_Studio\SignSpeak_Studio.exe"
    exit /b 0
)

if exist "SignSpeak_Studio.exe" (
    start "" "SignSpeak_Studio.exe"
    exit /b 0
)

echo Launching with Python...
python prototype\part_3_letters.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with code %ERRORLEVEL%.
    pause
)
