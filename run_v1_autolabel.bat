@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_v1_autolabel.bat DATASET_ROOT [OUTPUT_JSONL]
  exit /b 2
)
set "OUTPUT=%~2"
if "%OUTPUT%"=="" set "OUTPUT=.\artifacts\v1_predictions\predictions.jsonl"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_v1_autolabel.ps1" -DatasetRoot "%~1" -Output "%OUTPUT%"
exit /b %ERRORLEVEL%
