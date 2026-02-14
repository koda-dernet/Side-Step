@echo off
REM ====================================================================
REM  Side-Step Easy Installer for Windows
REM  Double-click this file or run it from a terminal.
REM ====================================================================
echo.
echo   ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
echo   ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
echo   ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
echo        ██ ██ ██   ██ ██                 ██    ██    ██      ██
echo   ███████ ██ ██████  ███████       ███████    ██    ███████ ██
echo.
echo   Standalone Installer (v0.8.0-beta)
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] PowerShell is required but was not found.
    echo        Windows 10/11 should have it built-in.
    pause
    exit /b 1
)

REM Run the PowerShell installer with execution policy bypass
powershell -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] Installation encountered errors. Check the output above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Press any key to close...
pause >nul
