@echo off
setlocal
set "REPO_ROOT=%~dp0..\.."
if not defined PYTHON_EXE set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%~dp0run_uav_08BU.py" --repo-root "%REPO_ROOT%" %*
exit /b %errorlevel%
