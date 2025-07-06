# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re
import tempfile

import os
import shutil
import sys
import time
import threading
import string

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
import gdown
import csv
import py7zr
import pathlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/sovits_svc")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/rvc")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/infer")

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# sovits
from sovits_svc import svc_inference

from sovits_svc.hubert.inference import hubert_infer
from sovits_svc.pitch.inference import pitch_infer
from sovits_svc.whisper.inference import whisper_infer

from omegaconf import OmegaConf
from huggingface_hub import snapshot_download
from sovits_svc.vits.models import SynthesizerInfer
from sovits_svc.pitch import load_csv_pitch

# applio
from rvc.infer.infer import VoiceConverter

# RVC
from infer.configs.config import Config
from infer.modules.vc.modules import VC

rvc_config = Config()
rvc_vc = VC(rvc_config)

cur_rvc_model_path = ""
custom_ema_model, pre_custom_path = None, ""

# load models

vocoder = load_vocoder()


def get_drive_id(url):
    """ 通过网盘文件url获取id """
    pattern = r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:file/d/|folder/d/|open\?id=|uc\?id=|drive/folders/)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return url


def load_custom(model_name: str, lang: str, password="", model_cfg=None, show_info=gr.Info):
    global F5_models_dict
    model_dict_list = F5_models_dict.get(lang, [])
    model_dict = None
    for model in model_dict_list:
        if model["model_name"] == model_name:
            model_dict = model
            break
    if model_dict is None:
        print("指定的模型不存在")
        return None

    # model_dict
    ckpt_path = model_dict["model"]
    vocab_path = model_dict["vocab"]
    if not os.path.exists(vocab_path):
        # 如果模型不存在，就根据链接下载
        model_url = model_dict["model_url"]
        if model_url != "":
            show_info("下载模型中……")
            file_id = get_drive_id(model_url)
            download_folder = "./F5-models/" + lang
            download_path = download_folder + "/" + model_name + ".7z"
            os.makedirs(download_folder, exist_ok=True)
            if not os.path.exists(download_path):
                gdown.download(id=file_id, output=download_path, fuzzy=True)
            # 解压
            if password == "":
                print("密码为空，请设置解压密码")
                gr.Warning("密码为空，请设置解压密码")
                return None
            try:
                show_info("解压模型中……")
                with py7zr.SevenZipFile(download_path, 'r', password=password) as archive:
                    archive.extractall(path=download_folder)
                os.remove(download_path)
            except Exception as e:
                print(str(e))
                show_info("模型解压失败")
                return None

            # 获取model路径
            model_folder = download_folder + "/" + model_name
            for model_file in os.listdir(model_folder):
                if model_file.lower().endswith(".safetensors") or model_file.lower().endswith(".pt"):
                    model_dict["model"] = model_folder + "/" + model_file
                    ckpt_path = model_dict["model"]
                elif model_file.lower().endswith(".txt"):
                    vocab_path = model_folder + "/" + model_file
                    model_dict["vocab"] = vocab_path

        if not os.path.exists(vocab_path):
            print("模型不存在")
            return None

    if model_cfg is None:
        if "v1" in os.path.basename(ckpt_path).lower():
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        else:
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False,
                             conv_layers=4, pe_attn_head=1)

    global pre_custom_path, custom_ema_model
    if pre_custom_path != ckpt_path or custom_ema_model is None:
        custom_ema_model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)
        pre_custom_path = ckpt_path

    return custom_ema_model


def load_F5_models_from_csv():
    csv_path = "./F5-models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 名称	语言	model链接
        for row in reader:
            model_name = row['名称'].strip()
            model_lang = row['语言'].strip()
            model_url = row['model链接'].strip()
            if model_name == "" or model_lang == "" or model_url == "":
                continue

            model_path = ""
            vocab_path = ""
            model_folder = "./F5-models/" + model_lang + "/" + model_name
            if os.path.exists(model_folder):
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".safetensors") or file.lower().endswith(".pt"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".txt"):
                        vocab_path = model_folder + "/" + file

            model_dict = {}
            model_dict["model_name"] = model_name
            model_dict["model"] = model_path
            model_dict["vocab"] = vocab_path
            model_dict["model_url"] = model_url
            model_dict["lang"] = model_lang
            if model_lang in models_dict:
                models_dict[model_lang].append(model_dict)
            else:
                models_dict[model_lang] = [model_dict]

    return models_dict


def load_F5_models_list():
    models_root_path = "./F5-models"
    models_dict = {}
    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./F5-models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载TODO
        models = load_F5_models_from_csv()
        models_dict.update(models)
        return models_dict

    if not os.path.exists(models_root_path):
        return models_dict

    # 第一层是语种
    for language in os.listdir(models_root_path):
        lang_folder = models_root_path + "/" + language
        if os.path.isdir(lang_folder):
            # 第二层是模型名
            for model_name in os.listdir(lang_folder):
                # 第三层是具体的模型文件
                model_folder = lang_folder + "/" + model_name
                vocab_file = model_folder + "/vocab.txt"
                if not os.path.exists(vocab_file):
                    continue

                for model_file in os.listdir(model_folder):
                    if model_file.lower().endswith(".safetensors") or model_file.lower().endswith(".pt"):
                        model_path = model_folder + "/" + model_file

                        model_dict = {}
                        model_dict["model_name"] = model_name
                        model_dict["model"] = model_path
                        model_dict["vocab"] = vocab_file
                        model_dict["model_url"] = ""
                        model_dict["lang"] = language
                        if language in models_dict:
                            models_dict[language].append(model_dict)
                        else:
                            models_dict[language] = [model_dict]

                        break

    return models_dict


def load_sovits_models_from_csv():
    csv_path = "./sovits-models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 音色名称	语种	模型链接
        for row in reader:
            model_name = row['音色名称'].strip()
            model_lang = row['语种'].strip()
            model_url = row['模型链接'].strip()
            if model_name == "" or model_lang == "" or model_url == "":
                continue

            model_path = ""
            speaker_path = ""
            model_folder = "./sovits-models/" + model_lang + "/" + model_name
            if os.path.exists(model_folder):
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".npy"):
                        speaker_path = model_folder + "/" + file

            # model_path = "./sovits-models/" + model_lang + "/" + model_name + "/sovits5.0.pth"
            # speaker_path = "./sovits-models/" + model_lang + "/" + model_name + "/speaker0.spk.npy"
            model_dict = {}
            model_dict["model_name"] = model_name
            model_dict["model_path"] = model_path.replace("\\", "/")
            model_dict["speaker_path"] = speaker_path.replace("\\", "/")
            model_dict["model_url"] = model_url

            if model_lang in models_dict:
                models_dict[model_lang].append(model_dict)
            else:
                models_dict[model_lang] = [model_dict]

    return models_dict


def load_sovits_models_list():
    models_root_path = "./sovits-models"
    models_dict = {}

    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./sovits-models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载TODO
        models = load_sovits_models_from_csv()
        models_dict.update(models)
        return models_dict

    if not os.path.exists(models_root_path):
        return models_dict
    for folder in os.listdir(models_root_path):
        folder_path = models_root_path + "/" + folder
        if os.path.isdir(folder_path):
            language = folder
            for model in os.listdir(folder_path):
                model_path = ""
                speaker_path = ""
                model_folder = folder_path + "/" + model
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".npy"):
                        speaker_path = model_folder + "/" + file

                if model_path != "" and speaker_path != "":
                    model_dict = {}
                    model_dict["model_name"] = model
                    model_dict["model_path"] = model_path.replace("\\", "/")
                    model_dict["speaker_path"] = speaker_path.replace("\\", "/")
                    model_dict["model_url"] = ""

                    if language in models_dict:
                        models_dict[language].append(model_dict)
                    else:
                        models_dict[language] = [model_dict]

    return models_dict


def load_applio_models_from_csv():
    csv_path = "./applio-models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 音色名称	语种	模型链接
        for row in reader:
            model_name = row['音色名称'].strip()
            model_lang = row['语种'].strip()
            model_url = row['模型链接'].strip()
            if model_name == "" or model_lang == "" or model_url == "":
                continue

            model_path = ""
            index_path = ""
            model_folder = "./applio-models/" + model_lang + "/" + model_name
            if os.path.exists(model_folder):
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".index"):
                        index_path = model_folder + "/" + file

            model_dict = {}
            model_dict["model_name"] = model_name
            model_dict["model_path"] = model_path.replace("\\", "/")
            model_dict["index_path"] = index_path.replace("\\", "/")
            model_dict["model_url"] = model_url

            if model_lang in models_dict:
                models_dict[model_lang].append(model_dict)
            else:
                models_dict[model_lang] = [model_dict]

    return models_dict


def load_applio_models_list():
    models_root_path = "./applio-models"
    models_dict = {}

    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./applio-models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载
        models = load_applio_models_from_csv()
        models_dict.update(models)
        return models_dict

    if not os.path.exists(models_root_path):
        return models_dict
    for language in os.listdir(models_root_path):
        language_folder = models_root_path + "/" + language
        if os.path.isdir(language_folder):
            for model in os.listdir(language_folder):
                model_path = ""
                index_path = ""
                model_folder = language_folder + "/" + model
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".index"):
                        index_path = model_folder + "/" + file

                if model_path != "" and index_path != "":
                    model_dict = {}
                    model_dict["model_name"] = model
                    model_dict["model_path"] = model_path.replace("\\", "/")
                    model_dict["index_path"] = index_path.replace("\\", "/")
                    model_dict["model_url"] = ""

                    if language in models_dict:
                        models_dict[language].append(model_dict)
                    else:
                        models_dict[language] = [model_dict]

    return models_dict


def load_rvc_models_from_csv():
    csv_path = "./rvc-models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 音色名称	语种	模型链接
        for row in reader:
            model_name = row['音色名称'].strip()
            model_lang = row['语种'].strip()
            model_url = row['模型链接'].strip()
            if model_name == "" or model_lang == "" or model_url == "":
                continue

            model_path = ""
            index_path = ""
            model_folder = "./rvc-models/" + model_lang + "/" + model_name
            if os.path.exists(model_folder):
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".index"):
                        index_path = model_folder + "/" + file

            model_dict = {}
            model_dict["model_name"] = model_name
            model_dict["model_path"] = model_path.replace("\\", "/")
            model_dict["index_path"] = index_path.replace("\\", "/")
            model_dict["model_url"] = model_url

            if model_lang in models_dict:
                models_dict[model_lang].append(model_dict)
            else:
                models_dict[model_lang] = [model_dict]

    return models_dict


def load_rvc_models_list():
    models_root_path = "./rvc-models"
    models_dict = {}

    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./rvc-models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载
        models = load_rvc_models_from_csv()
        models_dict.update(models)
        return models_dict

    if not os.path.exists(models_root_path):
        return models_dict
    for language in os.listdir(models_root_path):
        language_folder = models_root_path + "/" + language
        if os.path.isdir(language_folder):
            for model in os.listdir(language_folder):
                model_path = ""
                index_path = ""
                model_folder = language_folder + "/" + model
                for file in os.listdir(model_folder):
                    if file.lower().endswith(".pth"):
                        model_path = model_folder + "/" + file
                    elif file.lower().endswith(".index"):
                        index_path = model_folder + "/" + file

                if model_path != "" and index_path != "":
                    model_dict = {}
                    model_dict["model_name"] = model
                    model_dict["model_path"] = model_path.replace("\\", "/")
                    model_dict["index_path"] = index_path.replace("\\", "/")
                    model_dict["model_url"] = ""

                    if language in models_dict:
                        models_dict[language].append(model_dict)
                    else:
                        models_dict[language] = [model_dict]

    return models_dict


def load_refs_list():
    refs_root_path = "./refs"
    refs_dict = {}
    if not os.path.exists(refs_root_path):
        # 如果refs文件夹不存在，那就到网盘下载一份，这里需要实现网盘下载refs文件夹的功能。TODO
        return refs_dict

    for folder in os.listdir(refs_root_path):
        folder_path = refs_root_path + "/" + folder
        if os.path.isdir(folder_path):
            language = folder

            ref_audio_dict = {}
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for file in filenames:
                    if file.lower().endswith(".wav") or file.lower().endswith(".mp3"):
                        ref_audio_path = dirpath + "/" + file
                        ref_txt_path = ref_audio_path[:-4] + ".txt"
                        if os.path.exists(ref_txt_path):
                            ref_audio_dict[ref_audio_path.replace("\\", "/")] = ref_txt_path.replace("\\", "/")

            refs_dict[language] = ref_audio_dict

    return refs_dict


# 组合生成的音频，并在中间根据参数添加静音
def get_final_wave(cross_fade_duration, generated_waves, final_sample_rate):
    final_wave = None
    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * final_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    return final_wave


F5_models_dict = load_F5_models_list()
sovits_models_dict = load_sovits_models_list()
applio_models_dict = load_applio_models_list()
rvc_models_dict = load_rvc_models_list()
refs_dict = load_refs_list()
launch_in_service = False


def get_sovits_model(svc_model, lang_alone, password, show_info=gr.Info):
    global sovits_models_dict
    cur_speaker = None
    model_path = ""
    speaker_path = ""
    if lang_alone in sovits_models_dict:
        svc_models_list = sovits_models_dict[lang_alone]
        for speaker in svc_models_list:
            if speaker["model_name"] == svc_model:
                model_path = speaker["model_path"]
                speaker_path = speaker["speaker_path"]
                model_url = speaker["model_url"]
                cur_speaker = speaker
                break

    if model_path == "" or (not os.path.exists(model_path)):
        if model_url != "":
            show_info("下载sovits模型中……")
            file_id = get_drive_id(model_url)
            download_folder = "./sovits-models/" + lang_alone
            download_path = download_folder + "/" + svc_model + ".7z"
            os.makedirs(download_folder, exist_ok=True)
            if not os.path.exists(download_path):
                gdown.download(id=file_id, output=download_path, fuzzy=True)
            # 解压
            if password == "":
                print("密码为空，请设置解压密码")
                gr.Warning("密码为空，请设置解压密码")
                return False, None, None
            try:
                show_info("解压sovits模型中……")
                with py7zr.SevenZipFile(download_path, 'r', password=password) as archive:
                    archive.extractall(path=download_folder)

                # 解压完成后模型路径
                model_folder = download_folder + "/" + svc_model
                if os.path.exists(model_folder):
                    for file in os.listdir(model_folder):
                        if file.lower().endswith(".pth"):
                            model_path = os.path.abspath(model_folder + "/" + file)
                        elif file.lower().endswith(".npy"):
                            speaker_path = os.path.abspath(model_folder + "/" + file)

                os.remove(download_path)
            except Exception as e:
                print(str(e))
                show_info("sovits模型解压失败")
                return False, None, None

    if not os.path.exists(model_path):
        print("sovits模型不存在")
        gr.Warning("sovits模型不存在，无法继续")
        return False, "", ""
    else:
        cur_speaker["model_path"] = model_path
        cur_speaker["speaker_path"] = speaker_path
        return True, model_path, speaker_path


def get_applio_model(svc_model, lang_alone, password, show_info=gr.Info):
    global applio_models_dict
    cur_speaker = None
    model_path = ""
    index_path = ""
    model_url = ""
    if lang_alone in applio_models_dict:
        svc_models_list = applio_models_dict[lang_alone]
        for speaker in svc_models_list:
            if speaker["model_name"] == svc_model:
                model_path = speaker["model_path"]
                index_path = speaker["index_path"]
                model_url = speaker["model_url"]
                cur_speaker = speaker
                break

    if model_path == "" or (not os.path.exists(model_path)):
        if model_url != "":
            show_info("下载Applio模型中……")
            file_id = get_drive_id(model_url)
            download_folder = "./applio-models/" + lang_alone
            download_path = download_folder + "/" + svc_model + ".7z"
            os.makedirs(download_folder, exist_ok=True)
            if not os.path.exists(download_path):
                gdown.download(id=file_id, output=download_path, fuzzy=True)
            # 解压
            if password == "":
                print("密码为空，请设置解压密码")
                gr.Warning("密码为空，请设置解压密码")
                return False, None, None
            try:
                show_info("解压Applio模型中……")
                with py7zr.SevenZipFile(download_path, 'r', password=password) as archive:
                    archive.extractall(path=download_folder)

                # 解压完成后模型路径
                model_folder = download_folder + "/" + svc_model
                if os.path.exists(model_folder):
                    for file in os.listdir(model_folder):
                        if file.lower().endswith(".pth"):
                            model_path = model_folder + "/" + file
                        elif file.lower().endswith(".index"):
                            index_path = model_folder + "/" + file

                os.remove(download_path)
            except Exception as e:
                print(str(e))
                show_info("Applio模型解压失败")
                return False, None, None

    if not os.path.exists(model_path):
        print("Applio模型不存在")
        gr.Warning("Applio模型不存在，无法继续")
        return False, "", ""
    else:
        cur_speaker["model_path"] = model_path
        cur_speaker["index_path"] = index_path
        return True, model_path, index_path


def get_rvc_model(svc_model, lang_alone, password, show_info=gr.Info):
    global rvc_models_dict
    cur_speaker = None
    model_path = ""
    index_path = ""
    model_url = ""
    if lang_alone in rvc_models_dict:
        svc_models_list = rvc_models_dict[lang_alone]
        for speaker in svc_models_list:
            if speaker["model_name"] == svc_model:
                model_path = speaker["model_path"]
                index_path = speaker["index_path"]
                model_url = speaker["model_url"]
                cur_speaker = speaker
                break

    if model_path == "" or (not os.path.exists(model_path)):
        if model_url != "":
            show_info("下载RVC模型中……")
            file_id = get_drive_id(model_url)
            download_folder = "./rvc-models/" + lang_alone
            download_path = download_folder + "/" + svc_model + ".7z"
            os.makedirs(download_folder, exist_ok=True)
            if not os.path.exists(download_path):
                gdown.download(id=file_id, output=download_path, fuzzy=True)
            # 解压
            if password == "":
                print("密码为空，请设置解压密码")
                gr.Warning("密码为空，请设置解压密码")
                return False, None, None
            try:
                show_info("解压RVC模型中……")
                with py7zr.SevenZipFile(download_path, 'r', password=password) as archive:
                    archive.extractall(path=download_folder)

                # 解压完成后模型路径
                model_folder = download_folder + "/" + svc_model
                if os.path.exists(model_folder):
                    for file in os.listdir(model_folder):
                        if file.lower().endswith(".pth"):
                            model_path = model_folder + "/" + file
                        elif file.lower().endswith(".index"):
                            index_path = model_folder + "/" + file

                os.remove(download_path)
            except Exception as e:
                print(str(e))
                show_info("RVC模型解压失败")
                return False, None, None

    if not os.path.exists(model_path):
        print("RVC模型不存在")
        gr.Warning("RVC模型不存在，无法继续")
        return False, "", ""
    else:
        cur_speaker["model_path"] = model_path
        cur_speaker["index_path"] = index_path
        return True, model_path, index_path


def get_svc_model(enable_svc, svc_type, svc_model, lang_alone, password, show_info=gr.Info):
    if enable_svc:
        if svc_type is None or svc_model is None:
            return False, "", ""
        else:
            if svc_type == "Sovits":
                return get_sovits_model(svc_model, lang_alone, password, show_info=gr.Info)
            elif svc_type == "Applio":
                # Applio
                return get_applio_model(svc_model, lang_alone, password, show_info=gr.Info)
            elif svc_type == "RVC":
                # RVC
                return get_rvc_model(svc_model, lang_alone, password, show_info=gr.Info)


class sovits_parms():
    def __init__(self):
        self.config = ""
        self.model = ""
        self.wave = ""
        self.spk = ""
        self.ppg = None
        self.vec = None
        self.pit = None
        self.shift = 0
        self.pit_type = 'rmvpe'  # (sing or voice)
        self.enable_retrieval = False
        self.retrieval_index_prefix = ""
        self.retrieval_ratio = 0.5
        self.n_retrieval_vectors = 3
        self.hubert_index_path = ""
        self.whisper_index_path = ""
        self.debug = False
        self.voice = ""


def download_sovits_models():
    snapshot_download(
        repo_id="Jack202410/sovits-pretrain",
        local_dir='./',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,  # Allow downloading new files
        ignore_patterns=["*.git*"],  # Ignore git-related files
        resume_download=True  # Resume interrupted downloads
    )


def download_applio_models():
    snapshot_download(
        repo_id="Jack202410/applio-pretrain",
        local_dir='./rvc/models',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,  # Allow downloading new files
        ignore_patterns=["*.git*"],  # Ignore git-related files
        resume_download=True  # Resume interrupted downloads
    )


def download_rvc_models():
    snapshot_download(
        repo_id="Jack202410/rvc_pretrain2",
        local_dir='./infer/assets',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,  # Allow downloading new files
        ignore_patterns=["*.git*"],  # Ignore git-related files
        resume_download=True  # Resume interrupted downloads
    )


def sovits_convert_audio(audio_filepath, model_path, speaker_path, pitch=0):
    args = sovits_parms()
    args.model = model_path
    args.spk = speaker_path
    args.config = "./sovits_svc/configs/base.yaml"
    args.voice = svc_model
    args.wave = audio_filepath
    args.shift = pitch

    # 检查预训练模型是否存在，不存在就到网上下载
    whisper_pretrain = "./whisper_pretrain/large-v2.pt"
    if not os.path.exists(whisper_pretrain):
        download_sovits_models()

    temp_dir = os.path.join("temp", "temp_" + os.path.basename(args.wave))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    start_cleanup_thread("./temp", days=2)

    if (args.ppg == None):
        args.ppg = os.path.join(temp_dir, "svc_tmp.ppg.npy")
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        whisper_infer(args.wave, args.ppg)

    if (args.vec == None):
        args.vec = os.path.join(temp_dir, "svc_tmp.vec.npy")
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        hubert_infer(args.wave, args.vec)

    if (args.pit == None):
        args.pit = os.path.join(temp_dir, "svc_tmp.pit.csv")
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        pitch_infer(args.wave, args.pit, args.pit_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    svc_inference.load_svc_model(args.model, model)
    retrieval = svc_inference.create_retrival(args)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)
    # ppg = torch.zeros_like(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)
    # vec = torch.zeros_like(vec)

    pit = load_csv_pitch(args.pit)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    shift_info = ''
    if args.shift > 0:
        shift_info = "(+" + str(args.shift) + ")"
    elif args.shift < 0:
        shift_info = "(" + str(args.shift) + ")"
    out_audio = svc_inference.svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device, temp_dir)
    # out_file = os.path.join(temp_dir, f"{args.voice}{shift_info}-{os.path.basename(args.wave)}")
    # write(out_file, hp.data.sampling_rate, out_audio)
    return (hp.data.sampling_rate, out_audio)


def applio_convert_audio(audio_filepath, model_path, index_path, rvc_index_rate, pitch=0, split_audio=True):
    # 根据需要下载预训练模型
    rmvpe_pretrain = "./rvc/models/predictors/rmvpe.pt"
    if not os.path.exists(rmvpe_pretrain):
        download_applio_models()

    kwargs = {
        "audio_input_path": audio_filepath,
        "audio_output_path": "",
        "model_path": model_path,
        "index_path": index_path,
        "pitch": pitch,
        "index_rate": rvc_index_rate,
        "volume_envelope": 1,
        "protect": 0.5,
        "hop_length": 128,
        "f0_method": "rmvpe",
        "pth_path": model_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": False,
        "f0_autotune_strength": 1.0,
        "clean_audio": False,
        "clean_strength": 0.5,
        "export_format": "WAV",
        "f0_file": "",
        "embedder_model": "contentvec",
        "embedder_model_custom": None,
        "post_process": False,
        "formant_shifting": False,
        "formant_qfrency": 1.0,
        "formant_timbre": 1.0,
        "reverb": False,
        "pitch_shift": (pitch != 0),
        "limiter": False,
        "gain": False,
        "distortion": False,
        "chorus": False,
        "bitcrush": False,
        "clipping": False,
        "compressor": False,
        "delay": False,
        "reverb_room_size": 0.5,
        "reverb_damping": 0.5,
        "reverb_wet_level": 0.5,
        "reverb_dry_level": 0.5,
        "reverb_width": 0.5,
        "reverb_freeze_mode": 0.5,
        "pitch_shift_semitones": 0.0,
        "limiter_threshold": -6,
        "limiter_release": 0.01,
        "gain_db": 0.0,
        "distortion_gain": 25,
        "chorus_rate": 1.0,
        "chorus_depth": 0.25,
        "chorus_delay": 7,
        "chorus_feedback": 0.0,
        "chorus_mix": 0.5,
        "bitcrush_bit_depth": 8,
        "clipping_threshold": -6,
        "compressor_threshold": 0,
        "compressor_ratio": 1,
        "compressor_attack": 1.0,
        "compressor_release": 100,
        "delay_seconds": 0.5,
        "delay_feedback": 0.0,
        "delay_mix": 0.5,
        "sid": 0,
    }
    infer_pipeline = VoiceConverter()
    return infer_pipeline.convert_audio(**kwargs, )


def rvc_convert_audio(audio_filepath, model_path, index_path, rvc_index_rate, pitch=0, split_audio=True):
    # 根据需要下载预训练模型
    rmvpe_pretrain = "./infer/assets/rmvpe/rmvpe.pt"
    if not os.path.exists(rmvpe_pretrain):
        download_rvc_models()

    # 如果模型已经加载过了，就不再重复加载
    global cur_rvc_model_path
    if cur_rvc_model_path != model_path:
        rvc_vc.get_vc(model_path, 0.33)
        cur_rvc_model_path = model_path

    return rvc_vc.vc_single(0, audio_filepath, pitch, "", "rmvpe", index_path, "", rvc_index_rate, 3, 0, 0.25, 0.33,
                            split_audio)


def convert_audios(audios, language, svc_type, svc_model, tone_shift, rvc_index_rate, password, show_info=gr.Info):
    if len(audios) == 0:
        gr.Warning("请添加转换音频")
        return []
    if svc_model == "":
        gr.Warning("请选择转换模型")
        return []

    lang = language
    lang_alone = ""
    if "-" in language:
        index = language.find("-")
        lang = language[index + 1:]
        lang_alone = language[:index]

    enable_svc, model_path, speaker_path = get_svc_model(True, svc_type, svc_model, lang_alone, password, show_info)
    if model_path is None or model_path == "":
        gr.Warning("选择的音色模型不存在")
        return []

    # 删除旧的音频文件
    global launch_in_service
    if launch_in_service:
        os.makedirs("./convert_audio", exist_ok=True)
        convert_audio_path = tempfile.mkdtemp(dir="./convert_audio")
    else:
        convert_audio_path = "./convert_audio"
        if not os.path.exists(convert_audio_path):
            os.mkdir(convert_audio_path)
        else:
            # 把里面的东西删除
            for file in os.listdir(convert_audio_path):
                try:
                    os.remove(convert_audio_path + "/" + file)
                except:
                    pass

    svc_files = []
    progress = gr.Progress()
    for i, audio_filepath in enumerate(progress.tqdm(audios, desc="Processing")):
        file_name = pathlib.Path(audio_filepath).stem
        if svc_type == "Sovits":
            sampling_rate, audio_wave = sovits_convert_audio(audio_filepath, model_path, speaker_path, tone_shift)
            converted_filepath = convert_audio_path + f"/{file_name}_sovits_{tone_shift}.wav"
            sf.write(converted_filepath, audio_wave, sampling_rate, 'PCM_24')
            svc_files.append(converted_filepath)
        elif svc_type == "Applio":
            sampling_rate, audio_wave = applio_convert_audio(audio_filepath, model_path, speaker_path, rvc_index_rate,
                                                             tone_shift)
            if audio_wave is not None:
                converted_filepath = convert_audio_path + f"/{file_name}_applio_{tone_shift}_{rvc_index_rate}.wav"
                sf.write(converted_filepath, audio_wave, sampling_rate, 'PCM_24')
                svc_files.append(converted_filepath)
        elif svc_type == "RVC":
            sampling_rate, audio_wave = rvc_convert_audio(audio_filepath, model_path, speaker_path, rvc_index_rate,
                                                          tone_shift, True)
            if audio_wave is not None:
                converted_filepath = convert_audio_path + f"/{file_name}_rvc_{tone_shift}_{rvc_index_rate}.wav"
                sf.write(converted_filepath, audio_wave, sampling_rate, 'PCM_24')
                svc_files.append(converted_filepath)

    return svc_files


last_cleanup_times = {}
cleanup_lock = threading.Lock()


def delete_old_files_and_dirs(path, days=2):
    global last_cleanup_times
    now = time.time()
    with cleanup_lock:
        last_time = last_cleanup_times.get(path, 0)
        if (now - last_time) < 86400:
            return

        last_cleanup_times[path] = now

    cutoff = now - (days * 86400)
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        try:
            mtime = os.path.getmtime(full_path)
            if mtime < cutoff:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
        except Exception as e:
            print(f"[{path}] 处理时出错: {full_path}, 错误: {e}")


def start_cleanup_thread(path, days=2):
    t = threading.Thread(target=delete_old_files_and_dirs, args=(path, days), daemon=True)
    t.start()


PUNCT = set(string.punctuation + '，。！？；：（）【】《》、')


def insert_punct(text):
    result = []
    i = 0
    n = len(text)

    while i < n:
        if i + 1 < n and text[i] == ' ' and text[i + 1] == ' ':
            # 连续两个空格
            prev = result[-1] if result else ''
            if prev not in PUNCT:
                result.append('.')  # 插入句号
                result.append(' ')  # 插入空格
            # 跳过两个空格
            i += 2
        elif text[i] == ' ':
            # 单个空格
            prev = result[-1] if result else ''
            if prev not in PUNCT:
                result.append(',')  # 插入逗号
                result.append(' ')  # 插入空格
            # 跳过空格
            i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def infer(
        ref_audio_orig,
        ref_text,
        num_input,
        gen_texts,
        language,
        model_name,
        password,
        remove_silence,
        seed,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1,
        show_info=gr.Info,
        save_line_audio=False,
        insert_punct_in_space=False,
        enable_svc=True,
        svc_type="",
        svc_model="",
        tone_shift=0,
        rvc_index_rate=0.75,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), []

    # Set inference seed
    if seed < 0 or seed > 2 ** 31 - 1:
        gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
        seed = np.random.randint(0, 2 ** 31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    lang = language
    lang_alone = ""
    if "-" in language:
        index = language.find("-")
        lang = language[index + 1:]
        lang_alone = language[:index]

    show_info("加载模型……")
    ema_model = load_custom(model_name, language, password)
    if ema_model is None:
        gr.Warning("模型加载失败")
        return gr.update(), []

    # 检查用户设置的输入框数量是否正常
    try:
        num_input = int(num_input)
        if num_input <= 0 or num_input > 20:
            gr.Warning("输入框数量设置不对")
            return gr.update(), []
    except ValueError:
        gr.Warning("输入框数量无法转换为数字")
        return gr.update(), []

    # 把所有的输入框的文本都获取出来，汇总到一块，生成时也用总的这个列表，这样方便显示总进度
    # 如果要根据框保存中间结果，那就在生成的过程中判断是第几个框，所以保存每个框里的文本行的数量。
    all_gen_text_list = []
    text_box_text_list = []
    for i in range(num_input):
        gen_text_box = gen_texts[i]
        if gen_text_box.strip() == "":
            continue
        index = 0
        gen_text_list = gen_text_box.split("\n")
        for gen_text in gen_text_list:
            if gen_text.strip() == "":
                continue
            if (lang_alone == "泰语" and insert_punct_in_space) or "泰语-sit-男-4" in model_name:
                gen_text = insert_punct(gen_text)

            all_gen_text_list.append(gen_text)
            index += 1

        text_box_text_list.append(index)

    if len(text_box_text_list) == 0:
        gr.Warning("没有要生成的文本")
        return gr.update(), []

    # 删除旧的音频文件
    global launch_in_service
    if launch_in_service:
        os.makedirs("./gen_audio", exist_ok=True)
        gen_audio_path = tempfile.mkdtemp(dir="./gen_audio")
        start_cleanup_thread("./gen_audio", days=2)

        os.makedirs("./last_audio", exist_ok=True)
        last_audio_path = tempfile.mkdtemp(dir="./last_audio")
        start_cleanup_thread("./last_audio", days=2)

        os.makedirs("./tmp", exist_ok=True)
        tmp_audio_path = tempfile.mkdtemp(dir="./tmp")
        start_cleanup_thread("./tmp", days=2)
    else:
        gen_audio_path = "./gen_audio"
        if not os.path.exists(gen_audio_path):
            os.mkdir(gen_audio_path)
        else:
            # 把里面的东西删除
            for file in os.listdir(gen_audio_path):
                try:
                    os.remove(gen_audio_path + "/" + file)
                except:
                    pass
        last_audio_path = "./last_audio"
        if not os.path.exists(last_audio_path):
            os.mkdir(last_audio_path)
        else:
            # 把里面的东西删除
            for file in os.listdir(last_audio_path):
                try:
                    if file.endswith(".wav"):
                        os.remove(last_audio_path + "/" + file)
                except:
                    pass
        tmp_audio_path = "./tmp"
        if not os.path.exists(tmp_audio_path):
            os.mkdir(tmp_audio_path)
        else:
            # 把里面的东西删除
            for file in os.listdir(tmp_audio_path):
                try:
                    if file.endswith(".wav"):
                        os.remove(tmp_audio_path + "/" + file)
                except:
                    pass

    # 如果开启了转换功能，那就判断转换模型是否支持改语种，如果不支持就把转换功能关闭
    # 如果支持，就判断模型是否存在，如果不存在，就到网盘下载
    if enable_svc:
        enable_svc, model_path, speaker_path = get_svc_model(enable_svc, svc_type, svc_model, lang_alone, password,
                                                             show_info)
        if model_path is None:
            return gr.update(), []

    # 开始生成
    generated_waves = []
    svc_waves = []
    spectrograms = []
    progress = gr.Progress()
    start_pos = 0
    cur_box_index = 1
    total_box_count = text_box_text_list[0]
    svc_sampling_rate = 32000
    segm_audio_list = []
    for i, gen_text in enumerate(progress.tqdm(all_gen_text_list, desc="Processing")):

        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text.lower(),
            gen_text.lower(),
            ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=show_info,
            # progress=gr.Progress(),
            # lang=lang,
        )

        generated_waves.append(final_wave)
        spectrograms.append(combined_spectrogram)

        # 把音频转换改为一句一转换，也就是每生成一句音频，就需要在本地保存一下，并进行转换
        audio_filepath = tmp_audio_path + f"/tmp_audio_{i + 1}.wav"

        # 保存中间结果
        if save_line_audio:
            # 按行保存
            audio_filepath = gen_audio_path + f"/{model_name}_segm_audio_{i + 1}.wav"
            sf.write(audio_filepath, final_wave, final_sample_rate, 'PCM_24')
            segm_audio_list.append(audio_filepath)

        if enable_svc:
            if not os.path.exists(audio_filepath):
                sf.write(audio_filepath, final_wave, final_sample_rate, 'PCM_24')
            if svc_type == "Sovits":
                sampling_rate, audio_wave = sovits_convert_audio(audio_filepath, model_path, speaker_path, tone_shift)
                svc_waves.append(audio_wave)
                svc_sampling_rate = sampling_rate
            elif svc_type == "Applio":
                sampling_rate, audio_wave = applio_convert_audio(audio_filepath, model_path, speaker_path,
                                                                 rvc_index_rate, tone_shift, False)
                if audio_wave is not None:
                    svc_waves.append(audio_wave)
                    svc_sampling_rate = sampling_rate
            elif svc_type == "RVC":
                sampling_rate, audio_wave = rvc_convert_audio(audio_filepath, model_path, speaker_path, rvc_index_rate,
                                                              tone_shift, False)
                if audio_wave is not None:
                    svc_waves.append(audio_wave)
                    svc_sampling_rate = sampling_rate
                else:
                    print("转换失败-----")

        if not save_line_audio:
            # 按文本框保存，需要根据每个文本框的文本行数判断有没有到当前框的结尾，然后进行保存。
            if i == total_box_count - 1:
                final_waves = get_final_wave(cross_fade_duration, generated_waves[start_pos:], final_sample_rate)
                audio_filepath = gen_audio_path + f"/{model_name}_segm_audio_{cur_box_index}.wav"
                sf.write(audio_filepath, final_waves, final_sample_rate, 'PCM_24')
                segm_audio_list.append(audio_filepath)

                if cur_box_index < len(text_box_text_list):
                    start_pos = total_box_count
                    total_box_count += text_box_text_list[cur_box_index]
                    cur_box_index += 1

    output_audio_list = []
    if enable_svc:
        # 导出合并后的24Khz音频
        last_orgi_audio_path = last_audio_path + f"/{model_name}_orgi_audio.wav"
        final_waves = None
        if len(generated_waves) > 0:
            final_waves = get_final_wave(cross_fade_duration, generated_waves, final_sample_rate)
            sf.write(last_orgi_audio_path, final_waves, final_sample_rate, 'PCM_24')
            output_audio_list.append(last_orgi_audio_path)

        # 导出转换后音频
        last_gen_audio_path = last_audio_path + f"/{svc_model}_{svc_type.lower()}_audio.wav"
        final_waves = None
        if len(svc_waves) > 0:
            final_waves = get_final_wave(cross_fade_duration, svc_waves, svc_sampling_rate)
            sf.write(last_gen_audio_path, final_waves, svc_sampling_rate, 'PCM_24')
            final_sample_rate = svc_sampling_rate
            output_audio_list.append(last_gen_audio_path)
    else:
        # 导出合并后的24Khz音频
        last_gen_audio_path = last_audio_path + f"/{model_name}_orgi_audio.wav"
        final_waves = None
        if len(generated_waves) > 0:
            final_waves = get_final_wave(cross_fade_duration, generated_waves, final_sample_rate)
            sf.write(last_gen_audio_path, final_waves, final_sample_rate, 'PCM_24')
            output_audio_list.append(last_gen_audio_path)

    # Remove silence
    if remove_silence and os.path.exists(last_gen_audio_path):
        remove_silence_for_generated_wav(last_gen_audio_path)
        final_waves, _ = torchaudio.load(last_gen_audio_path)
        final_waves = final_waves.squeeze().cpu().numpy()

    output_audio_list.extend(segm_audio_list)
    return (final_sample_rate, final_waves), output_audio_list, used_seed


def create_textboxes(num):
    try:
        num = int(num)
        if num <= 0:
            return [gr.update(visible=False) for _ in range(20)]

        # 控制输入框的可见性，最多支持 20 个
        updates = [gr.update(visible=True) if i < num else gr.update(visible=False) for i in range(20)]
        return updates
    except ValueError:
        return [gr.update(visible=False) for _ in range(20)]


def clear_txt_boxs():
    return [gr.update(value="") for _ in range(20)]


def load_ref_txt(ref_txt_path):
    txt = ""
    if os.path.exists(ref_txt_path):
        with open(ref_txt_path, "r", encoding="utf8") as f:
            txt = f.read()
    return txt


with gr.Blocks(title="F5-TTS-SVC_v3") as app:
    gr.Markdown(
        """
# 自定义 F5 TTS + SVC

F5-TTS + SOVITS + Applio + RVC

"""
    )


    def get_default_params(lang):
        global F5_models_dict
        global refs_dict
        model_dict_list = F5_models_dict.get(lang, [])
        model_names = []
        for model_dict in model_dict_list:
            model_names.append(model_dict["model_name"])
        def_model = ""
        if len(model_names) > 0:
            def_model = model_names[0]

        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        ref_audios = []
        ref_txts = []
        for key, value in ref_audio_dict.items():
            ref_audios.append(key)
            ref_txts.append(value)
        if len(ref_audios) > 0 and os.path.exists(ref_audios[0]):
            def_audio = ref_audios[0]
        else:
            def_audio = None

        if len(ref_txts) > 0:
            def_txt = load_ref_txt(ref_txts[0]).strip()
        else:
            def_txt = ""

        speaker_models = []
        speaker_model = None
        svc_type_list = []
        def_svc_type = None

        global applio_models_dict
        global rvc_models_dict
        global sovits_models_dict
        if lang_alone in applio_models_dict:
            svc_type_list.append("Applio")
        if lang_alone in sovits_models_dict:
            svc_type_list.append("Sovits")
        if lang_alone in rvc_models_dict:
            svc_type_list.append("RVC")

        # 优先看看该语种有没有RVC模型，如果有就优先用RVC
        if "Applio" in svc_type_list:
            def_svc_type = "Applio"
        elif "Sovits" in svc_type_list:
            def_svc_type = "Sovits"
        elif "RVC" in svc_type_list:
            def_svc_type = "RVC"

        if def_svc_type == "Applio":
            svc_models_list = applio_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])
        elif def_svc_type == "Sovits":
            svc_models_list = sovits_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])
        elif def_svc_type == "RVC":
            svc_models_list = rvc_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])

        if len(speaker_models) > 0:
            speaker_model = speaker_models[0]

        return (model_names, def_model), def_txt, (ref_audios, def_audio), (svc_type_list, def_svc_type), (
            speaker_models, speaker_model)


    def language_change(lang):
        (model_names, def_model), def_txt, (ref_audios, def_audio), (svc_type_list, def_svc_type), (speaker_models,
                                                                                                    speaker_model) = get_default_params(
            lang)

        return gr.update(choices=model_names, value=def_model), gr.update(value=def_txt), \
            gr.update(choices=ref_audios, value=def_audio), \
            gr.update(choices=svc_type_list, value=def_svc_type), \
            gr.update(choices=speaker_models, value=speaker_model), \
            gr.update(interactive=(def_svc_type == "Applio" or def_svc_type == "RVC"))


    def ref_audio_change(lang, audio_path):
        global refs_dict
        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        def_txt = ""
        for key, value in ref_audio_dict.items():
            if key == audio_path:
                def_txt = load_ref_txt(value).strip()
                break

        return gr.update(value=def_txt)


    def svc_type_change(lang, svc_type):

        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        speaker_models = []
        speaker_model = None

        global applio_models_dict
        global rvc_models_dict
        global sovits_models_dict
        if svc_type == "Applio":
            svc_models_list = applio_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])
        elif svc_type == "Sovits":
            svc_models_list = sovits_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])
        elif svc_type == "RVC":
            svc_models_list = rvc_models_dict[lang_alone]
            for speaker in svc_models_list:
                speaker_models.append(speaker["model_name"])

        if len(speaker_models) > 0:
            speaker_model = speaker_models[0]

        return gr.update(choices=speaker_models, value=speaker_model), \
            gr.update(interactive=(svc_type == "Applio" or svc_type == "RVC"))


    if len(F5_models_dict) > 0:
        def_lang = list(F5_models_dict.keys())[0]
        (model_names, def_model), def_txt, (ref_audios, def_audio), \
            (svc_type_list, def_svc_type), (speaker_models, speaker_model) = get_default_params(def_lang)
    else:
        model_names = [],
        def_model = "",
        def_txt = "'",
        ref_audios = [],
        def_audio = "",
        svc_type_list = [],
        def_svc_type = None,
        speaker_models = [],
        speaker_model = None
        def_lang = ""

    with gr.Tabs():
        with gr.TabItem("F5-TTS + SVC"):
            with gr.Row():
                # 在这里添加新语言的支持，记得在languages里添加语言的英文对照
                language = gr.Dropdown(
                    choices=list(F5_models_dict.keys()), value=def_lang, label="语言", allow_custom_value=True
                )
                custom_ckpt_path = gr.Dropdown(
                    choices=model_names,
                    value=def_model,
                    allow_custom_value=True,
                    label="模型",
                    visible=True,
                )
                ref_audio = gr.Dropdown(
                    choices=ref_audios, value=def_audio, label="参考音频", allow_custom_value=True
                )

            with gr.Row():
                enable_svc = gr.Checkbox(
                    label="启用音频转换",
                    info="勾选此项，启动音频转换。",
                    value=True,
                )

                svc_type = gr.Dropdown(
                    choices=svc_type_list,
                    value=def_svc_type,
                    label="音频转换音色选择",
                    visible=True,
                )

                svc_model = gr.Dropdown(
                    choices=speaker_models,
                    value=speaker_model,
                    label="音频转换音色选择",
                    visible=True,
                )

                password_input = gr.Textbox(
                    label="解压密码:",
                    type="password",
                    lines=1,
                    value="",
                )

            with gr.Row():
                num_input = gr.Textbox(label="请输入需要的输入框数量(1-20)", value="1", scale=1)
                tone_shift_slider = gr.Slider(
                    label="音调调整",
                    minimum=-12,
                    maximum=12,
                    value=0,
                    step=1,
                    info="音调偏移",
                    scale=2
                )
                rvc_index_rate_slider = gr.Slider(
                    label="语搜索特征比率",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                    info="索引文件施加的影响;值越高，影响越大。但是，选择较低的值有助于减少音频中存在的伪影。",
                    interactive=(def_svc_type == "Applio" or def_svc_type == "RVC"),
                    scale=2
                )

            # 动态布局区域
            rows = []
            max_per_row = 5
            textboxes = []

            # 创建一个动态布局，最多 20 个输入框
            for i in range(4):  # 每行最多 5 个，4 行总共 20 个
                with gr.Row() as row:
                    for j in range(max_per_row):
                        index = i * max_per_row + j
                        if index == 0:
                            textbox = gr.Textbox(label=f"生成文本:{index + 1}", lines=10, visible=True)
                        else:
                            textbox = gr.Textbox(label=f"生成文本:{index + 1}", lines=10, visible=False)
                        textboxes.append(textbox)
                    rows.append(row)

            num_input.change(create_textboxes, inputs=[num_input], outputs=textboxes)

            modify_words_input = gr.Textbox(label="改词", lines=10)

            with gr.Row():
                clear_box_btn = gr.Button("清空文本框", variant="primary", scale=0.2)
                generate_btn = gr.Button("合成", variant="primary")
                download_all = gr.Button("下载所有输出音频", variant="primary")

            with gr.Accordion("高级设置", open=False):
                basic_ref_text_input = gr.Textbox(
                    label="参考音频对应文本",
                    info="如果留空则自动转录生成. 如果输入文本，则使用输入的文本，建议输入标准文本，转录出来的文本准确性可能不高。",
                    lines=2,
                    value=def_txt,
                )
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="随机种子",
                        info="勾选此项，每次会自动生成随机值",
                        value=True,
                        scale=1,
                    )
                    seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
                    remove_silence = gr.Checkbox(
                        label="删除静音",
                        info="模型会自动生成静音，尤其是短音频，通过此选项可以移除静音。",
                        value=False,
                    )
                    save_line_audio = gr.Checkbox(
                        label="按行保存音频",
                        info="勾选此项，中间结果会每行保存一个音频，不勾选，则每一个文本框保存一个音频。",
                        value=False,
                    )
                    insert_punct_in_space = gr.Checkbox(
                        label="插入标点",
                        info="此功能针对泰语，在空格处插入逗号",
                        value=False,
                    )
                speed_slider = gr.Slider(
                    label="语速设置",
                    minimum=0.3,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    info="调整音频语速",
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=64,
                    step=2,
                    info="Set the number of denoising steps.",
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    info="设置两个片段之前的静音时长",
                )

            audio_output = gr.Audio(label="合成音频")
            download_output = gr.File(label="下载文件", file_count="multiple")

            language.change(
                language_change,
                inputs=[language],
                outputs=[custom_ckpt_path, basic_ref_text_input, ref_audio, svc_type, svc_model, rvc_index_rate_slider],
                show_progress="hidden",
            )
            ref_audio.change(
                ref_audio_change,
                inputs=[language, ref_audio],
                outputs=[basic_ref_text_input],
                show_progress="hidden",
            )
            svc_type.change(
                svc_type_change,
                inputs=[language, svc_type],
                outputs=[svc_model, rvc_index_rate_slider],
                show_progress="hidden",
            )

            def basic_tts(
                    ref_audio_input,
                    ref_text_input,
                    language,
                    model_name,
                    password,
                    remove_silence,
                    randomize_seed,
                    seed_input,
                    save_line_audio,
                    insert_punct_in_space,
                    cross_fade_duration_slider,
                    nfe_slider,
                    speed_slider,
                    enable_svc,
                    svc_type,
                    svc_model,
                    tone_shift,
                    rvc_index_rate,
                    num_input,
                    modify_words,
                    *gen_texts_input,
            ):
                if randomize_seed:
                    seed_input = np.random.randint(0, 2 ** 31 - 1)

                if len(modify_words) > 0:
                    gen_texts_input_modify = []
                    for text in gen_texts_input:
                        for modify in modify_words.split("\n"):
                            parts = modify.split("\t")
                            if len(parts) != 2:
                                continue
                            else:
                                text = text.replace(parts[0], parts[1])
                                gen_texts_input_modify.append(text)
                else:
                    gen_texts_input_modify = gen_texts_input

                audio_out, gen_audio_list, used_seed = infer(
                    ref_audio_input,
                    ref_text_input,
                    num_input,
                    gen_texts_input_modify,
                    language,
                    model_name,
                    password,
                    remove_silence,
                    seed=seed_input,
                    cross_fade_duration=cross_fade_duration_slider,
                    nfe_step=nfe_slider,
                    speed=speed_slider,
                    save_line_audio=save_line_audio,
                    insert_punct_in_space=insert_punct_in_space,
                    enable_svc=enable_svc,
                    svc_type=svc_type,
                    svc_model=svc_model,
                    tone_shift=tone_shift,
                    rvc_index_rate=rvc_index_rate,
                )
                return audio_out, gen_audio_list, used_seed


            intputs = [ref_audio,
                       basic_ref_text_input,
                       language,
                       custom_ckpt_path,
                       password_input,
                       remove_silence,
                       randomize_seed,
                       seed_input,
                       save_line_audio,
                       insert_punct_in_space,
                       cross_fade_duration_slider,
                       nfe_slider,
                       speed_slider,
                       enable_svc,
                       svc_type,
                       svc_model,
                       tone_shift_slider,
                       rvc_index_rate_slider,
                       num_input,
                       modify_words_input
                       ] + textboxes

            generate_btn.click(
                basic_tts,
                inputs=intputs,
                outputs=[audio_output, download_output, seed_input],
            )

            clear_box_btn.click(
                clear_txt_boxs,
                outputs=textboxes,
            )

            download_all.click(None, [], [], js="""
                () => {
                    const component = Array.from(document.getElementsByTagName('label')).find(el => el.textContent.trim() === '下载文件').parentElement;
                    const links = component.getElementsByTagName('a');
                    for (let link of links) {
                        if (link.href.startsWith("http:") && !link.href.includes("127.0.0.1")) {
                            link.href = link.href.replace("http:", "https:");
                        }
                        link.click();
                    }
                }
            """)

        with gr.TabItem("音色转换"):
            def convert_language_change(lang):

                lang_alone = lang
                if "-" in lang_alone:
                    index = lang.find("-")
                    lang_alone = lang_alone[:index]

                speaker_models = []
                speaker_model = None
                svc_type_list = []
                def_svc_type = None

                global applio_models_dict
                global rvc_models_dict
                global sovits_models_dict
                if lang_alone in applio_models_dict:
                    svc_type_list.append("Applio")
                if lang_alone in sovits_models_dict:
                    svc_type_list.append("Sovits")
                if lang_alone in rvc_models_dict:
                    svc_type_list.append("RVC")

                # 优先看看该语种有没有RVC模型，如果有就优先用RVC
                if "Applio" in svc_type_list:
                    def_svc_type = "Applio"
                elif "Sovits" in svc_type_list:
                    def_svc_type = "Sovits"
                elif "RVC" in svc_type_list:
                    def_svc_type = "RVC"

                if def_svc_type == "Applio":
                    svc_models_list = applio_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])
                elif def_svc_type == "Sovits":
                    svc_models_list = sovits_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])
                elif def_svc_type == "RVC":
                    svc_models_list = rvc_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])

                if len(speaker_models) > 0:
                    speaker_model = speaker_models[0]

                return gr.update(choices=svc_type_list, value=def_svc_type), \
                    gr.update(choices=speaker_models, value=speaker_model), \
                    gr.update(interactive=(def_svc_type == "Applio" or def_svc_type == "RVC"))


            def convert_svc_type_change(lang, svc_type):

                lang_alone = lang
                if "-" in lang_alone:
                    index = lang.find("-")
                    lang_alone = lang_alone[:index]

                speaker_models = []
                speaker_model = None

                global applio_models_dict
                global rvc_models_dict
                global sovits_models_dict
                if svc_type == "Applio":
                    svc_models_list = applio_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])
                elif svc_type == "Sovits":
                    svc_models_list = sovits_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])
                elif svc_type == "RVC":
                    svc_models_list = rvc_models_dict[lang_alone]
                    for speaker in svc_models_list:
                        speaker_models.append(speaker["model_name"])

                if len(speaker_models) > 0:
                    speaker_model = speaker_models[0]

                return gr.update(choices=speaker_models, value=speaker_model), \
                    gr.update(interactive=(svc_type == "Applio" or svc_type == "RVC"))


            def scanning_segm_audios():
                gen_audio_path = "./gen_audio"
                audios = []
                for file in os.listdir(gen_audio_path):
                    audios.append(gen_audio_path + "/" + file)

                return audios


            with gr.Row():
                convert_language = gr.Dropdown(
                    choices=list(F5_models_dict.keys()), value=def_lang, label="语言", allow_custom_value=True
                )

                convert_svc_type = gr.Dropdown(
                    choices=svc_type_list,
                    value=def_svc_type,
                    label="音频转换音色选择",
                    visible=True,
                )

                convert_svc_model = gr.Dropdown(
                    choices=speaker_models,
                    value=speaker_model,
                    label="音频转换音色选择",
                    visible=True,
                )

                convert_password = gr.Textbox(
                    label="解压密码:",
                    type="password",
                    lines=1,
                    value="",
                )

            with gr.Row():
                convert_tone_shift_slider = gr.Slider(
                    label="音调调整",
                    minimum=-12,
                    maximum=12,
                    value=0,
                    step=1,
                    info="音调偏移",
                )
                convert_index_rate_slider = gr.Slider(
                    label="语搜索特征比率",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                    info="索引文件施加的影响;值越高，影响越大。但是，选择较低的值有助于减少音频中存在的伪影。",
                    interactive=(def_svc_type == "Applio" or def_svc_type == "RVC"),
                )
            with gr.Row():
                input_convert_audios = gr.File(label="转换音频", file_count="multiple")
                output_audios = gr.File(label="输出音频", file_count="multiple")

            with gr.Row():
                convert_btn = gr.Button("转换", variant="primary")
                segm_audio_btn = gr.Button("填充分段音频", variant="primary")
                download_outputs = gr.Button("下载所有输出音频", variant="primary")

            convert_language.change(
                convert_language_change,
                inputs=[convert_language],
                outputs=[convert_svc_type, convert_svc_model, convert_index_rate_slider],
                show_progress="hidden",
            )
            convert_svc_type.change(
                convert_svc_type_change,
                inputs=[convert_language, convert_svc_type],
                outputs=[convert_svc_model, convert_index_rate_slider],
                show_progress="hidden",
            )

            convert_btn.click(
                convert_audios,
                inputs=[input_convert_audios, convert_language, convert_svc_type, convert_svc_model, \
                        convert_tone_shift_slider, convert_index_rate_slider, convert_password],
                outputs=[output_audios],
            )

            segm_audio_btn.click(
                scanning_segm_audios,
                outputs=[input_convert_audios],
            )

            download_outputs.click(None, [], [], js="""
                () => {
                    const component = Array.from(document.getElementsByTagName('label')).find(el => el.textContent.trim() === '输出音频').parentElement;
                    const links = component.getElementsByTagName('a');
                    for (let link of links) {
                        if (link.href.startsWith("http:") && !link.href.includes("127.0.0.1")) {
                            link.href = link.href.replace("http:", "https:");
                        }
                        link.click();
                    }
                }
            """)

        modify_words_input.change(None, modify_words_input, None, js="(v)=>{ setStorage('modifyWords', v) }")

        js_get_local_storage = """
                   						    function() {
                   						      globalThis.setStorage = (key, value)=>{
                   						        localStorage.setItem(key, JSON.stringify(value))
                   						      }
                   						       globalThis.getStorage = (key, value)=>{
                   						        return JSON.parse(localStorage.getItem(key))
                   						      }
                   						       var modifyWords = getStorage('modifyWords')
                   						       return [modifyWords];
                   						      }
                   						    """
        app.load(
            None,
            inputs=None,
            outputs=[modify_words_input],
            js=js_get_local_storage,
        )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=False, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
@click.option(
    "--inservice",
    "-s",
    is_flag=True,
    default=False,
    help="Launch in service",
)
def main(port, host, share, api, root_path, inbrowser, inservice):
    global app
    global launch_in_service
    launch_in_service = inservice
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    main()
