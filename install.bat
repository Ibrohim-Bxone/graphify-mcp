@echo off
chcp 65001 >nul
echo ============================================
echo   Graphify - Claude Code uchun doimiy xotira
echo   O'rnatish boshlandi...
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [XATO] Python topilmadi!
    echo Python 3.10+ o'rnating: https://www.python.org/downloads/
    echo O'rnatishda "Add python.exe to PATH" katagini belgilang!
    pause
    exit /b 1
)

echo [1/3] Python muhiti yaratilmoqda...
python -m venv "%~dp0.venv"
if errorlevel 1 (
    echo [XATO] venv yaratilmadi. Python versiyangizni tekshiring.
    pause
    exit /b 1
)

echo [2/3] Kutubxonalar o'rnatilmoqda (bir necha daqiqa olishi mumkin)...
"%~dp0.venv\Scripts\pip.exe" install -q -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [XATO] Kutubxonalar o'rnatilmadi. Internet aloqasini tekshiring.
    pause
    exit /b 1
)

echo [3/3] Claude Code'ga ulanmoqda...
"%~dp0.venv\Scripts\python.exe" "%~dp0setup.py"
if errorlevel 1 (
    echo [XATO] Sozlashda muammo chiqdi.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   TAYYOR! Graphify o'rnatildi.
echo   Endi Claude Code'ni yopib, qayta oching.
echo   Batafsil: README.md
echo ============================================
pause
