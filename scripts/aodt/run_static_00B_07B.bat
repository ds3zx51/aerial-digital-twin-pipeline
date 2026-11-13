@echo off
setlocal

if not defined AODT_PYTHON set "AODT_PYTHON=python"

set "SCRIPT_DIR=%~dp0"
"%AODT_PYTHON%" "%SCRIPT_DIR%run_static_00B_07B.py" %*
exit /b %ERRORLEVEL%
