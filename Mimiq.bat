@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" goto run
echo   Mimiq ещё не установлен - сначала запустите install.bat
pause
exit /b 1
:run
start "" ".venv\Scripts\pythonw.exe" -m mimiq %*
