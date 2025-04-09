import sys, os

import soundfile

from rmvpe.RMVPEF0Predictor import RMVPEF0Predictor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import librosa
import argparse
import numpy as np
import crepe
import parselmouth


def move_average(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


def compute_f0_mouth(path, device):
    # pip install praat-parselmouth

    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    lpad = 1024 // 160
    rpad = lpad
    f0 = parselmouth.Sound(x, sr).to_pitch_ac(
        time_step=160 / sr,
        voicing_threshold=0.5,
        pitch_floor=30,
        pitch_ceiling=1000).selected_array['frequency']
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    return f0


def compute_f0_salience(filename, device):
    from pitch.core.salience import salience
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    f0, t, s = salience(
        audio,
        Fs=sr,
        H=320,
        N=2048,
        F_min=45.0,
        F_max=1760.0)
    f0 = np.repeat(f0, 2, -1)  # 320 -> 160 * 2
    f0 = move_average(f0, 3)
    return f0


def compute_f0_voice(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = crepe.filter.mean(pitch, 3)
    pitch = pitch.squeeze(0)
    return pitch


def compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch


def compute_save_f0_rmvpe(wav, pit, device):
    predictor = RMVPEF0Predictor(hop_length=320, f0_min=50, f0_max=1100, device=device)
    audio, sampling_rate = soundfile.read(wav)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    pitch, uv = predictor.compute_f0_uv(audio)
    pitch = np.repeat(pitch, 2, -1)
    uv = np.repeat(uv, 2, -1)
    with open(pit, "w", encoding='utf-8') as pitch_file:
        pitch *= uv
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def pitch_infer(wav, pit, pit_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # rmvpe不管是唱歌还是说话都比较稳定
    if pit_type == 'crepe_sing':
        pitch = compute_f0_sing(wav, device)
        save_csv_pitch(pitch, pit)
    elif pit_type == 'crepe_voice':
        pitch = compute_f0_voice(wav, device)
        save_csv_pitch(pitch, pit)
    elif pit_type == 'parselmouth':
        pitch = compute_f0_mouth(wav, device)
        save_csv_pitch(pitch, pit)
    elif pit_type == 'salience':
        pitch = compute_f0_salience(wav, device)
        save_csv_pitch(pitch, pit)
    else:   # 默认使用rmvpe
        compute_save_f0_rmvpe(wav, pit, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--pit", help="pit", dest="pit", required=True)  # csv for excel
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    pitch_infer(args.wav, args.pit)
    # tmp = load_csv_pitch(args.pit)
    # save_csv_pitch(tmp, "tmp.csv")
