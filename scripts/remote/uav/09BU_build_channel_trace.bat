@echo off
setlocal
for %%I in ("%~dp0..\..\..") do set "REPO_ROOT=%%~fI"
if not defined AODT_PYTHON set "AODT_PYTHON=python"
"%AODT_PYTHON%" "%REPO_ROOT%\scripts\aodt\run_uav_09BU.py" --repo-root "%REPO_ROOT%" %*
exit /b %ERRORLEVEL%
