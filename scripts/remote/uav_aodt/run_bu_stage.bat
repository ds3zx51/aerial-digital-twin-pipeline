@echo off
setlocal
set "BU_SCRIPT_DIR=%~dp0"
if "%BU_STAGE%"=="" (
  echo ERROR: BU_STAGE was not set by the stage wrapper. 1>&2
  exit /b 2
)
set "BU_REPO_ROOT=%BU_SCRIPT_DIR%..\..\.."
if not defined BU_PYTHON if defined AODT_PYTHON set "BU_PYTHON=%AODT_PYTHON%"
if not defined BU_PYTHON set "BU_PYTHON=python"
"%BU_PYTHON%" "%BU_SCRIPT_DIR%..\..\aodt\bu_aerpaw\bu_pipeline.py" %BU_STAGE% --repo-root "%BU_REPO_ROOT%" %*
exit /b %ERRORLEVEL%
