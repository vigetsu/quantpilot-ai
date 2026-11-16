@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m streamlit run travel_photos\app.py --server.port 8502
) else (
  python -m streamlit run travel_photos\app.py --server.port 8502
)
pause
