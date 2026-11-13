@echo off
setlocal
set "REPO_ROOT=%~dp0..\.."
if not defined PYTHON_EXE set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%~dp0plot_static_08B.py" --repo-root "%REPO_ROOT%" %*
exit /b %errorlevel%
