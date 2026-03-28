@echo off
call .venv\Scripts\activate.bat
echo Starting DeepFake Lab with NVIDIA CUDA acceleration...
python run.py --execution-provider cuda
pause
