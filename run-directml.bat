@echo off
call .venv\Scripts\activate.bat
echo Starting DeepFake Lab with AMD DirectML acceleration...
python run.py --execution-provider dml
pause
