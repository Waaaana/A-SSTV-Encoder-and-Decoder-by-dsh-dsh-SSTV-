@echo off
rem ===================================================================
rem  SSTV Studio launcher
rem  Double-click this file, or run it from a command prompt.
rem
rem  Examples:
rem     run.bat                       open the window
rem     run.bat --check               verify the install, no window
rem     run.bat --decode capture.wav  decode a recording
rem
rem  NOTE: this file is deliberately ASCII-only.  cmd.exe reads .bat files
rem  using the OEM code page, not UTF-8, so non-ASCII text here would be
rem  parsed as commands.  The Chinese variant does the code-page switch
rem  first and only then prints Chinese.
rem ===================================================================
setlocal
cd /d "%~dp0"

set "WAIT="
if "%~1"=="" set "WAIT=1"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 sstv_app.py %*
    if errorlevel 1 set "WAIT=1"
    goto :finish
)

where python >nul 2>nul
if not errorlevel 1 (
    python sstv_app.py %*
    if errorlevel 1 set "WAIT=1"
    goto :finish
)

echo.
echo Python was not found on this computer.
echo.
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run this file again.
echo.
set "WAIT=1"

:finish
if defined WAIT pause
endlocal
exit /b 0