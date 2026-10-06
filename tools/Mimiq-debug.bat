@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Mimiq - отладка
echo   Mimiq в режиме отладки: все сообщения выводятся в это окно.
echo   Лог также пишется в %LOCALAPPDATA%\Mimiq\logs\mimiq.log
echo.
".venv\Scripts\python.exe" -X faulthandler -m mimiq %*
echo.
echo   Mimiq завершился с кодом %ERRORLEVEL%.
pause
