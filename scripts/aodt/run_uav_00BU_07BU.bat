@echo off
setlocal
if not defined AODT_PYTHON set "AODT_PYTHON=python"
set "SCRIPT_DIR=%~dp0"
"%AODT_PYTHON%" "%SCRIPT_DIR%run_uav_00BU_07BU.py" %*
exit /b %ERRORLEVEL%
