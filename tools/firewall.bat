@echo off
chcp 65001 >nul
rem Opens TCP 8443-8452 for Mimiq Link on private networks (asks for administrator rights).
net session >nul 2>nul
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
netsh advfirewall firewall delete rule name="Mimiq Link" >nul 2>nul
netsh advfirewall firewall add rule name="Mimiq Link" dir=in action=allow protocol=TCP localport=8443-8452 profile=private
echo.
echo   Готово: iPhone сможет подключаться к Mimiq в частной сети Wi-Fi.
echo   Если сеть в Windows отмечена как "Общедоступная", переключите её на "Частная":
echo   Параметры - Сеть и Интернет - Wi-Fi/Ethernet - Тип сетевого профиля - Частная.
pause
