@echo off
set PATH=%CD%\runtime\Scripts\;PATH=%CD%\runtime\;%PATH%
runtime\python.exe infer_gradio.py --inbrowser
pause
