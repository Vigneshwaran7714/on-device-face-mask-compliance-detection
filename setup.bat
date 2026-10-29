@echo off
REM One-time setup (Windows). Run by double-clicking.
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete.
pause
