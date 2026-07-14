@echo off
chcp 65001 >nul
echo Graphify o'chirilmoqda...
"%~dp0.venv\Scripts\python.exe" "%~dp0setup.py" --uninstall
echo.
echo Tayyor. Papkani (va xotira bazasi kg_db ni) xohlasangiz qo'lda o'chiring.
pause
