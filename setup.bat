@echo off

set PATH=%CD%\runtime\Scripts\;PATH=%CD%\runtime\;%PATH%
REM runtime\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
runtime\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
runtime\python.exe -m pip install -r requirements.txt
REM runtime\python.exe -m pip install gdown

echo ��װ����ɡ�
pause
