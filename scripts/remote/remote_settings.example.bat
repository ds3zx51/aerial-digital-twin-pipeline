@echo off

rem Copy this file to remote_settings.bat and edit ONLY that local copy.
rem remote_settings.bat is ignored by Git and must never be committed.

set "LOCAL_ROOT=C:\path\to\aerial-digital-twin-pipeline"

set "PLINK=C:\Program Files\PuTTY\plink.exe"
set "PSCP=C:\Program Files\PuTTY\pscp.exe"

rem AI Server SSH endpoint. This is NOT the AODT Worker gRPC endpoint.
set "REMOTE_USER=ubuntu"
set "REMOTE_HOST=CHANGE_ME_AI_SERVER_IP"
set "REMOTE_PORT=22"
set "REMOTE_ROOT=/path/to/aerial-digital-twin-pipeline"

rem Python inside the AdvTabDDPM virtual environment on the AI Server.
set "REMOTE_PYTHON=/path/to/aerial-digital-twin-pipeline/.venv-advtabddpm/bin/python"
set "REMOTE_ADVTABDDPM_RUNS_ROOT=/path/to/aerial-digital-twin-pipeline/advtabddpm_runs"
