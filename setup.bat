@echo off

set PATH=%CD%\runtime\Scripts\;PATH=%CD%\runtime\;%PATH%
runtime\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
runtime\python.exe -m pip install -r requirements.txt
runtime\python.exe -m pip install gdown
echo 安装包完成。
pause
