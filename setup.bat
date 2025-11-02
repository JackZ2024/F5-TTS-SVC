@echo off

set PATH=%CD%\runtime\Scripts\;%CD%\runtime\;%PATH%

runtime\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
runtime\python.exe -m pip install -r requirements.txt

echo 安装包完成。
pause
