@echo off
setlocal
call "%~dp0..\remote_settings.bat"
if errorlevel 1 (echo ERROR: copy scripts\remote\remote_settings.example.bat to remote_settings.bat first & exit /b 4)
if "%~1"=="" (echo Usage: 13B.bat ^<AODT_EVAL_RESULT_NAME^> & exit /b 1)
set "RESULT_NAME=%~1"
echo(%RESULT_NAME%| findstr /R /X "aodt_advtabddpm_eval_[A-Za-z0-9._-][A-Za-z0-9._-]*" >nul || exit /b 2
set "LOCAL_DIR=%LOCAL_ROOT%\results\%RESULT_NAME%"
if exist "%LOCAL_DIR%" (echo ERROR: refusing to overwrite %LOCAL_DIR% & exit /b 3)
mkdir "%LOCAL_DIR%" || exit /b %ERRORLEVEL%
"%PSCP%" -scp -P %REMOTE_PORT% "%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_ROOT%/advtabddpm_runs/packages/%RESULT_NAME%.tar.gz" "%LOCAL_DIR%\%RESULT_NAME%.tar.gz" || exit /b %ERRORLEVEL%
"%PSCP%" -scp -P %REMOTE_PORT% "%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_ROOT%/advtabddpm_runs/packages/%RESULT_NAME%.tar.gz.sha256" "%LOCAL_DIR%\%RESULT_NAME%.tar.gz.sha256"
exit /b %ERRORLEVEL%
