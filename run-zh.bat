@echo off
rem ===================================================================
rem  SSTV Studio - Chinese launcher (run-zh.bat)
rem
rem  Double-click to open the window, or use it from a command prompt:
rem     run-zh.bat                       open the window
rem     run-zh.bat --check               verify the install, no window
rem     run-zh.bat --decode REC.wav      decode a recording
rem
rem  IMPORTANT: cmd.exe parses .bat files with the OEM code page, not
rem  UTF-8.  Everything outside the echo messages below is ASCII on
rem  purpose; the messages themselves only work because the code page is
rem  switched to UTF-8 on the next line.
rem ===================================================================
setlocal
chcp 65001 >nul 2>nul
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
echo 没有找到 Python。
echo.
echo 请到 https://www.python.org/downloads/ 安装 Python 3.10 或更新版本，
echo 安装时记得勾选 "Add python.exe to PATH"，然后重新运行本文件。
echo.
set "WAIT=1"

:finish
if defined WAIT pause
endlocal
exit /b 0
