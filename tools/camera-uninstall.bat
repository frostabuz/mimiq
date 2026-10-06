@echo off
chcp 65001 >nul
rem Removes the "Mimiq Camera" virtual camera (asks for administrator rights).
net session >nul 2>nul
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set "MC=%ProgramW6432%\Mimiq Camera"
if not defined ProgramW6432 set "MC=%ProgramFiles%\Mimiq Camera"
if exist "%MC%\UnityCaptureFilter32.dll" if exist "%SystemRoot%\SysWOW64\regsvr32.exe" "%SystemRoot%\SysWOW64\regsvr32.exe" /s /u "%MC%\UnityCaptureFilter32.dll"
if exist "%MC%\UnityCaptureFilter64.dll" "%SystemRoot%\System32\regsvr32.exe" /s /u "%MC%\UnityCaptureFilter64.dll"
rmdir /s /q "%MC%" >nul 2>nul
echo.
echo   Готово: камера "Mimiq Camera" удалена.
echo   Если какое-то приложение ещё показывает её - перезапустите это приложение.
pause
