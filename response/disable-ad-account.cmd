@echo off
REM Wazuh Active Response launcher - pipes the alert JSON (stdin) into PowerShell.
REM Place next to disable-ad-account.ps1 in active-response\bin\ on DC01.
PowerShell.exe -ExecutionPolicy Bypass -NonInteractive -File "%~dp0disable-ad-account.ps1"
