@echo off
color 0A
cd /d "%~dp0"

echo ========================================================
echo        LAUNCHING RECEPTOR SIMULATOR (GUI)
echo ========================================================
echo.
echo Please enter the path to your python.exe (with all requirements installed).
echo (If your python is already in PATH or you are in an active environment, just press Enter)
echo.
set /p PYTHON_PATH="Path to python.exe [Press Enter for default]: "

if "%PYTHON_PATH%"=="" (
    set PYTHON_PATH=python
)

:: Remove quotes if the user accidentally pasted them, then add our own
set PYTHON_PATH=%PYTHON_PATH:"=%

echo.
echo Launching simulator using: "%PYTHON_PATH%"
echo ========================================================
echo.

"%PYTHON_PATH%" main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application crashed or Python was not found!
    echo Please make sure you specified the correct path to python.exe 
    echo and installed all dependencies via: pip install -r requirements.txt
    echo.
    pause
)
