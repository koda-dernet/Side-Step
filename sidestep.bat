@echo off
REM ====================================================================
REM  Side-Step Launcher (Windows)
REM
REM  Usage:
REM    sidestep.bat              — Interactive menu
REM    sidestep.bat gui           — Skip menu, launch GUI directly
REM    sidestep.bat train ...    — Skip menu, direct CLI mode
REM ====================================================================
cd /d "%~dp0"

where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

REM If args provided, pass through directly (CLI mode)
if not "%~1"=="" (
    uv run sidestep %*
    exit /b %ERRORLEVEL%
)

REM Interactive menu
echo.
echo   ░▒▓ S I D E · S T E P ▓▒░
echo   v1.0.0-beta
echo.
echo   [1]  Wizard   -- Interactive CLI training wizard
echo   [2]  GUI      -- Launch web GUI in browser
echo.
set /p choice="  Pick [1/2]: "

if "%choice%"=="2" (
    echo.
    echo   Launching GUI...
    uv run sidestep gui
) else (
    echo.
    echo   Launching wizard...
    uv run sidestep
)
