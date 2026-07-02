@echo off
cd /d "%~dp0"

set PYTHONPATH=%CD%\src;%CD%\scripts
set FIXED8_DATA_DIR=%CD%\outputs
set FIXED8_PRICE_CACHE_DIR=%CD%\work\price_cache

if "%FIXED8_API_TOKEN%"=="" (
  echo FIXED8_API_TOKEN is not set. API will run without a password on this computer/network.
  echo To protect it, run: set FIXED8_API_TOKEN=your-password
)

if exist C:\python\python.exe (
  C:\python\python.exe -m uvicorn api_app:app --host 0.0.0.0 --port 8000
) else (
  python -m uvicorn api_app:app --host 0.0.0.0 --port 8000
)

pause
