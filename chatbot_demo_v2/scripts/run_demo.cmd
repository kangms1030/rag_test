@echo off
rem ---------------------------------------------------------------
rem  chatbot_demo_v2 demo server  (Windows cmd)
rem    usage:  scripts\run_demo.cmd [port]      default port 8002
rem  NOTE: keep this file ASCII-only. cmd.exe reads batch files with
rem        the OEM codepage, so non-ASCII text here breaks parsing.
rem ---------------------------------------------------------------
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8002"

rem this script lives in ...\chatbot_demo_v2\scripts -> go up two levels
cd /d "%~dp0..\.."

call conda activate intern_chatbot
if errorlevel 1 (
    echo [ERROR] failed to activate conda env "intern_chatbot".
    echo         Run from Anaconda Prompt, or run "conda init cmd.exe" once.
    exit /b 1
)

echo.
echo   chatbot_demo_v2  :  http://127.0.0.1:%PORT%
echo   press Ctrl+C to stop
echo.
python -X utf8 -m chatbot_demo_v2 --port %PORT%
