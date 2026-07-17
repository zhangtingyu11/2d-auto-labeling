@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_v1_1_deploy.ps1" %*
exit /b %ERRORLEVEL%
