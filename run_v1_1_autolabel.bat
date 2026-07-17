@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_v1_1_autolabel.ps1" %*
exit /b %ERRORLEVEL%
