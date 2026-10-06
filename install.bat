@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Mimiq - установка

echo.
echo   Mimiq - установка
echo   -----------------
echo   Нужно ~4 ГБ на диске и интернет. Займёт 5-15 минут.
echo.

set "PY="
call :find_python
if defined PY goto have_python

echo   Python 3.12 не найден.
where winget >nul 2>nul
if errorlevel 1 goto no_python
choice /C YN /M "  Установить Python 3.12 автоматически через winget"
if errorlevel 2 goto no_python
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY goto have_python

:no_python
echo.
echo   Установите Python 3.12 с сайта python.org и включите галочку "Add python.exe to PATH".
echo   Потом снова запустите install.bat.
start "" "https://www.python.org/downloads/release/python-31210/"
pause
exit /b 1

:have_python
echo   Python: %PY%
if exist ".venv\Scripts\python.exe" goto venv_ok
echo   Создаю окружение .venv ...
%PY% -m venv .venv
if errorlevel 1 goto venv_fail

:venv_ok
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel --disable-pip-version-check -q
".venv\Scripts\python.exe" tools\install.py %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo   Установка не завершена - смотрите сообщения выше. Можно просто запустить install.bat ещё раз.
pause
exit /b %RC%

:venv_fail
echo   Не удалось создать окружение. Удалите папку .venv и попробуйте снова.
pause
exit /b 1

rem ---------------------------------------------------------------------------
:find_python
for %%V in (3.12 3.13 3.11) do (
  if not defined PY py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if defined PY exit /b 0
set "LP=%LOCALAPPDATA%\Programs\Python"
for %%P in ("%LP%\Python312\python.exe" "%ProgramFiles%\Python312\python.exe" "%LP%\Python313\python.exe" "%ProgramFiles%\Python313\python.exe" "%LP%\Python311\python.exe" "%ProgramFiles%\Python311\python.exe") do (
  if not defined PY if exist %%P set PY=%%P
)
if defined PY exit /b 0
python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul && set "PY=python"
exit /b 0
