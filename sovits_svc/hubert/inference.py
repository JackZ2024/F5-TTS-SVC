import sys, os

from whisper.audio import pad_or_trim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import librosa

from hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        N_SAMPLES = 2 * 20 * 16000
        feats = pad_or_trim(feats, N_SAMPLES, N_SAMPLES // 2)
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    np.save(vecPath, vec_a, allow_pickle=False)


hubert_m = None


def hubert_infer(wavPath, vecPath):
    global hubert_m
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hubert_m is None:
        hubert_m = load_model(os.path.join(
            "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)
    pred_vec(hubert_m, wavPath, vecPath, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    wavPath = args.wav
    vecPath = args.vec

    hubert_infer(wavPath, vecPath)
