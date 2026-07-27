@echo off
rem ---------------------------------------------------------------
rem  Expose the running demo through a Cloudflare Quick Tunnel.
rem    usage:  scripts\share_tunnel.cmd [port]     default port 8002
rem  Requires the demo server to be already running (run_demo.cmd).
rem  NOTE: keep this file ASCII-only (see run_demo.cmd).
rem ---------------------------------------------------------------
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8002"

where cloudflared >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cloudflared not found on PATH.
    echo         https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    exit /b 1
)

rem Make sure the local server answers first - otherwise the tunnel
rem opens fine but every visitor just gets a 502.
curl -s -o nul -m 3 "http://127.0.0.1:%PORT%/api/health"
if errorlevel 1 (
    echo [ERROR] nothing is answering on http://127.0.0.1:%PORT%
    echo         Start the server first, in another terminal:
    echo             chatbot_demo_v2\scripts\run_demo.cmd %PORT%
    exit /b 1
)

echo.
echo   local server OK  :  opening quick tunnel for port %PORT%
echo   share the https://....trycloudflare.com URL printed below.
echo   closing this window kills the link immediately.
echo.
cloudflared tunnel --url "http://127.0.0.1:%PORT%"
