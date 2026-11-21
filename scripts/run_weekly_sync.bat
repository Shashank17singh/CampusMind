@echo off
REM CampusMind - University of Lucknow Weekly Document Sync
cd /d "%~dp0\.."
echo [CampusMind] Running weekly document sync...
backend\.venv\Scripts\python.exe scripts\sync_uol_weekly.py
pause
