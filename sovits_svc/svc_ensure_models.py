import os.path
import shutil
import zipfile

import gdown
import wget


def ensure_models():
    if not os.path.exists("crepe/assets"):
        os.makedirs("crepe/assets")
    if not os.path.exists("crepe/assets/full.pth"):
        wget.download("https://github.com/maxrmorrison/torchcrepe/raw/master/torchcrepe/assets/full.pth",
                      out="crepe/assets/full.pth")

    if not os.path.exists("hubert_pretrain"):
        os.makedirs("hubert_pretrain")
    if not os.path.exists("hubert_pretrain/hubert-soft-0d54a1f4.pt"):
        wget.download("https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt",
                      "hubert_pretrain/hubert-soft-0d54a1f4.pt")

    if not os.path.exists("speaker_pretrain"):
        os.makedirs("speaker_pretrain")
    if not os.path.exists("speaker_pretrain/best_model.pth.tar"):
        gdown.download("https://drive.google.com/file/d/1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E/view?usp=drive_link",
                       "speaker_pretrain/best_model.pth.tar", fuzzy=True)

    if not os.path.exists("speaker_pretrain/config.json"):
        wget.download(
            "https://raw.githubusercontent.com/PlayVoice/so-vits-svc-5.0/bigvgan-mix-v2/speaker_pretrain/config.json",
            "speaker_pretrain/config.json")

    if not os.path.exists("vits_pretrain"):
        os.makedirs("vits_pretrain")
    if not os.path.exists("vits_pretrain/sovits5.0.pretrain.pth"):
        wget.download("https://github.com/PlayVoice/so-vits-svc-5.0/releases/download/5.0/sovits5.0.pretrain.pth",
                      "vits_pretrain/sovits5.0.pretrain.pth")

    if not os.path.exists("whisper_pretrain"):
        os.makedirs("whisper_pretrain")
    if not os.path.exists("whisper_pretrain/large-v2.pt"):
        wget.download(
            "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
            "whisper_pretrain/large-v2.pt")

    if not os.path.exists("rmvpe_pretrain"):
        os.makedirs("rmvpe_pretrain")
    if not os.path.exists("rmvpe_pretrain/rmvpe2.pt"):
        wget.download("https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip", "rmvpe.zip")
        with zipfile.ZipFile("rmvpe.zip", "r") as zip_ref:
            zip_ref.extractall("rmvpe_pretrain")
            shutil.move("rmvpe_pretrain/model.pt", "rmvpe_pretrain/rmvpe2.pt")
        os.remove("rmvpe.zip")