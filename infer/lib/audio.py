import platform, os
# import ffmpeg
import numpy as np
# import av
from io import BytesIO
import traceback
import re

import librosa
import soundfile as sf


# def wav2(i, o, format):
#     inp = av.open(i, "rb")
#     if format == "m4a":
#         format = "mp4"
#     out = av.open(o, "wb", format=format)
#     if format == "ogg":
#         format = "libvorbis"
#     if format == "mp4":
#         format = "aac"

#     ostream = out.add_stream(format)

#     for frame in inp.decode(audio=0):
#         for p in ostream.encode(frame):
#             out.mux(p)

#     for p in ostream.encode(None):
#         out.mux(p)

#     out.close()
#     inp.close()

def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return np.array(audio).flatten()

# def load_audio(file, sr):
#     try:
#         # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
#         # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#         # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#         file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
#         if os.path.exists(file) == False:
#             raise RuntimeError(
#                 "You input a wrong audio path that does not exists, please fix it!"
#             )
#         out, _ = (
#             ffmpeg.input(file, threads=0)
#             .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
#             .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
#         )
#     except Exception as e:
#         traceback.print_exc()
#         raise RuntimeError(f"Failed to load audio: {e}")

#     return np.frombuffer(out, np.float32).flatten()

def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(r'[\u202a\u202b\u202c\u202d\u202e]', '', path_str)  # 移除 Unicode 控制字符
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
