@echo off
echo ========================================
echo  Trading Bot Dashboard
echo ========================================
echo.
echo Installing dependencies...
pip install flask flask-cors -q

echo.
echo Starting server...
echo Access: http://localhost:5000
echo.
python server.py
pause
