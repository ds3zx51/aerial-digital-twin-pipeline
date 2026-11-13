@echo off
setlocal
call "%~dp0..\remote_settings.bat"
if errorlevel 1 (echo ERROR: copy scripts\remote\remote_settings.example.bat to remote_settings.bat first & exit /b 4)
if "%~1"=="" (echo Usage: 11B.bat ^<STATIC_RUN_NAME^> & exit /b 1)
set "RUN_NAME=%~1"
echo(%RUN_NAME%| findstr /R /X "[A-Za-z0-9][A-Za-z0-9_-]*" >nul || exit /b 2
set "SOURCE_NAME=aodt_dataset_static_%RUN_NAME%"
set "METRICS=%LOCAL_ROOT%\results\aodt_static\%RUN_NAME%\10B_colosseum_adapter\metrics_csv"
set "ARCHIVE=%LOCAL_ROOT%\results\aodt_static\%RUN_NAME%\10B_colosseum_adapter\%SOURCE_NAME%_metrics.zip"
if not exist "%METRICS%" (echo ERROR: passing 10B metrics are required & exit /b 10)
if exist "%ARCHIVE%" (echo ERROR: refusing to overwrite %ARCHIVE% & exit /b 11)
powershell -NoProfile -Command "Compress-Archive -LiteralPath (Get-ChildItem -LiteralPath '%METRICS%' -Filter '*_metrics.csv' | Select-Object -ExpandProperty FullName) -DestinationPath '%ARCHIVE%' -ErrorAction Stop" || exit /b %ERRORLEVEL%
"%PLINK%" -ssh -P %REMOTE_PORT% -l %REMOTE_USER% %REMOTE_HOST% "test -f %REMOTE_ROOT%/ai_server/AdvTabDDPM/AdvTabDDPM.py && test -f %REMOTE_ROOT%/ai_server/deploy/materialize_aodt_dataset.py && test -x %REMOTE_PYTHON% && mkdir -p %REMOTE_ROOT%/incoming %REMOTE_ROOT%/advtabddpm_runs/packages && test ! -e %REMOTE_ROOT%/incoming/%SOURCE_NAME%_metrics.zip && test ! -e %REMOTE_ROOT%/results/%SOURCE_NAME%" || exit /b %ERRORLEVEL%
"%PSCP%" -scp -P %REMOTE_PORT% "%ARCHIVE%" "%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_ROOT%/incoming/%SOURCE_NAME%_metrics.zip" || exit /b %ERRORLEVEL%
"%PLINK%" -ssh -P %REMOTE_PORT% -l %REMOTE_USER% %REMOTE_HOST% "AODT_EXPECTED_UE_COUNT=50 AODT_REMOTE_ROOT='%REMOTE_ROOT%' ADVTABDDPM_PYTHON='%REMOTE_PYTHON%' AODT_ADVTABDDPM_RUNS_ROOT='%REMOTE_ADVTABDDPM_RUNS_ROOT%' bash %REMOTE_ROOT%/ai_server/deploy/aodt_advtabddpm_remote.sh prepare %SOURCE_NAME% %REMOTE_ROOT%/incoming/%SOURCE_NAME%_metrics.zip"
exit /b %ERRORLEVEL%
