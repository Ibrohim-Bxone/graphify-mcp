@echo off
chcp 65001 >nul
echo Graphify dashboard ochilmoqda: http://localhost:5000
start "" http://localhost:5000
"%~dp0.venv\Scripts\python.exe" "%~dp0dashboard.py"
