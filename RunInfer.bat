@echo off
set PATH=%CD%\runtime\Scripts\;%CD%\runtime\;%PATH%

runtime\python.exe infer_gradio.py --inbrowser

pause
