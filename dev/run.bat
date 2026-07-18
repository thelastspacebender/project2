@echo off
color 0A
cd /d "%~dp0"
echo ========================================================
echo        LAUNCHING RECEPTOR SIMULATOR (GUI)
echo ========================================================
call "C:\Users\TheLastSpaceBender\Downloads\micromamba-win-64\Library\bin\micromamba.exe" run -r "C:\Users\TheLastSpaceBender\Downloads\micromamba-win-64\Library\bin" -n AIProgram python main.py
pause
