@echo off
echo Starting WiFi-DensePose API Backend...
set PYTHONIOENCODING=utf-8
set PYTHONPATH=archive\v1
call .venv\Scripts\activate.bat
uvicorn archive.v1.src.api.main:app --host 0.0.0.0 --port 8000
