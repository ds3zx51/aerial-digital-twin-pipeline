@echo off
setlocal
call "%~dp0..\remote_settings.bat"
if errorlevel 1 (echo ERROR: copy scripts\remote\remote_settings.example.bat to remote_settings.bat first & exit /b 4)
if "%~2"=="" (echo Usage: 12BU.bat ^<train^|eval^> ^<AODT_DATASET_NAME^> & exit /b 1)
set "ACTION=%~1"
set "SOURCE_NAME=%~2"
if /I not "%ACTION%"=="train" if /I not "%ACTION%"=="eval" exit /b 2
echo(%SOURCE_NAME%| findstr /R /X "aodt_dataset_uav_[A-Za-z0-9._-][A-Za-z0-9._-]*" >nul || exit /b 3
"%PLINK%" -ssh -P %REMOTE_PORT% -l %REMOTE_USER% %REMOTE_HOST% "AODT_EXPECTED_UE_COUNT=54 AODT_REMOTE_ROOT='%REMOTE_ROOT%' ADVTABDDPM_PYTHON='%REMOTE_PYTHON%' AODT_ADVTABDDPM_RUNS_ROOT='%REMOTE_ADVTABDDPM_RUNS_ROOT%' bash %REMOTE_ROOT%/ai_server/deploy/aodt_advtabddpm_remote.sh %ACTION% %SOURCE_NAME%"
exit /b %ERRORLEVEL%
