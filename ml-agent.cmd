@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3 "%SCRIPT_DIR%ml_agent.py" %*
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%SCRIPT_DIR%ml_agent.py" %*
    exit /b %ERRORLEVEL%
)

echo Python 3.10 or later is required.
echo Install Python from https://www.python.org/downloads/windows/
exit /b 1
