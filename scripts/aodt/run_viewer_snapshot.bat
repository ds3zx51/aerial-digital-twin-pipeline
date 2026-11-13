@echo off
setlocal

if not defined AODT_PYTHON set "AODT_PYTHON=python"

set "SCRIPT_DIR=%~dp0"
"%AODT_PYTHON%" "%SCRIPT_DIR%run_viewer_snapshot.py" %*
exit /b %ERRORLEVEL%
