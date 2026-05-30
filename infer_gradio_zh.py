# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import collections
import csv
import json
import os
import pathlib
import re
import shutil
import string
import sys
import tempfile
import threading
import time

import click
import gdown
import gradio as gr
import librosa
import numpy as np
import py7zr
import soundfile as sf
import torch
import torchaudio

import asr_sherpaonnx
from f5_tts.model.utils import convert_char_to_pinyin, preload_pypinyin_dicts
from third_party.text.g2pw import preload_pp_dicts


from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)



class _ModelCache:
    """
    LRU GPU 模型缓存。最多将 gpu_slots 个模型同时保留在显存中。
    超限时将空闲（不在推理中）的最旧模型卸载到 CPU RAM，
    再次请求时优先从 CPU RAM 恢复（比磁盘加载快得多）。
    _busy set 保证推理中的模型绝不会被驱逐。
    """

    def __init__(self, gpu_slots: int):
        self.gpu_slots = gpu_slots
        self._cond = threading.Condition()
        self._gpu: dict = {}                        # path -> model (on GPU)
        self._gpu_lru = collections.OrderedDict()  # LRU 顺序
        self._cpu: dict = {}                        # path -> model (on CPU RAM)
        self._busy: set = set()                     # 当前正在推理中的模型路径

    def _log_status(self, event: str):
        """打印操作事件、GPU/CPU 模型列表及剩余显存。"""
        gpu_names = [os.path.basename(p) for p in self._gpu]
        cpu_names = [os.path.basename(p) for p in self._cpu]
        if torch.cuda.is_available():
            free_bytes = torch.cuda.mem_get_info()[0]
            vram_info = f"{free_bytes / 1024 ** 3:.1f} GB free"
        else:
            vram_info = "N/A (no CUDA)"
        print(
            f"[ModelCache] {event}\n"
            f"  GPU({len(gpu_names)}/{self.gpu_slots}): {gpu_names}\n"
            f"  CPU cache({len(cpu_names)}):  {cpu_names}\n"
            f"  VRAM remaining: {vram_info}"
        )

    def acquire(self, ckpt_path: str, load_fn):
        """
        获取指定路径的模型（保证在 GPU 上），并标记为 busy。
        推理结束后必须调用 release(ckpt_path)。
        """
        with self._cond:
            # 已在 GPU，直接复用，更新 LRU
            if ckpt_path in self._gpu:
                self._busy.add(ckpt_path)
                self._gpu_lru.move_to_end(ckpt_path)
                return self._gpu[ckpt_path]

            # GPU 槽位已满，驱逐最旧的空闲模型到 CPU
            while len(self._gpu) >= self.gpu_slots:
                evicted = next(
                    (p for p in self._gpu_lru if p not in self._busy), None
                )
                if evicted is not None:
                    mdl = self._gpu.pop(evicted)
                    del self._gpu_lru[evicted]
                    mdl.cpu()
                    self._cpu[evicted] = mdl
                    torch.cuda.empty_cache()
                    self._log_status(f"GPU → CPU RAM: {os.path.basename(evicted)}")
                    break
                # 闂佸湱閸嬫捇鏌?GPU 濠碘檧鍋撳畷鍥ｅ亾閻戣姤鐒鹃柦妯洪煬鏌熼幁鎺戝闁癸攻缁嬪妲愬┑鍫㈤┏濠㈣泛锕︾粣锟犳煛鐏炴儳濮冮梺鍙夌矌閳ф嚀閺堝垂?
                self._cond.wait()

            # 婵?CPU 缂傚倸鍊归幐鎼佹偤閵娾晛绠ｉ柡宥嗗剮婵℃惈闁?
            if ckpt_path in self._cpu:
                mdl = self._cpu.pop(ckpt_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                mdl.to(device)
                self._log_status(f"CPU RAM → GPU: {os.path.basename(ckpt_path)}")
            else:
                mdl = load_fn()
                if mdl is None:
                    return None
                self._log_status(f"磁盘 → GPU: {os.path.basename(ckpt_path)}")

            self._gpu[ckpt_path] = mdl
            self._gpu_lru[ckpt_path] = None
            self._busy.add(ckpt_path)
            return mdl

    def release(self, ckpt_path: str):
        """推理完成，从 busy 中移除，并通知等待中的驱逐请求。"""
        with self._cond:
            self._busy.discard(ckpt_path)
            self._cond.notify_all()

# load models

vocoder = load_vocoder()


def preload_shared_pinyin_dicts():
    pypinyin_count = preload_pypinyin_dicts()
    g2pw_count = preload_pp_dicts()
    print(f"预加载词典：pypinyin={pypinyin_count}, g2pw={g2pw_count}")


preload_shared_pinyin_dicts()


def get_drive_id(url):
    """ 通过网盘文件url获取id """
    pattern = r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:file/d/|folder/d/|open\?id=|uc\?id=|drive/folders/)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return url


def get_model_dict(model_name: str, lang: str):
    model_dict_list = F5_models_dict.get(lang, [])
    for model in model_dict_list:
        if model["model_name"] == model_name:
            return model
    return None


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _parse_simple_config(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            key = key.strip().strip("\"'")
            value = value.strip().strip("\"'")
            if "#" in value:
                value = value.split("#", 1)[0].strip().strip("\"'")
            data[key] = value
    return data


def _read_pinyin_config_file(model_folder):
    config_names = [
        "pinyin_config.json",
        "pinyin_config.yaml",
        "pinyin_config.yml",
        "pinyin_config.toml",
        "config.json",
        "config.yaml",
        "config.yml",
        "config.toml",
    ]
    for config_name in config_names:
        config_path = os.path.join(model_folder, config_name)
        if not os.path.exists(config_path):
            continue
        if config_path.endswith(".json"):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return _parse_simple_config(config_path)
    return {}


def get_model_pinyin_config(model_name: str, lang: str):
    model_dict = get_model_dict(model_name, lang)
    if model_dict is None:
        return {"use_g2pw": True, "dict_name": None}

    pinyin_config = model_dict.get("pinyin_config")
    if pinyin_config is not None:
        return pinyin_config

    model_path = model_dict.get("model", "")
    model_folder = os.path.dirname(model_path) if model_path else model_dict.get("model_folder", "")
    config_data = _read_pinyin_config_file(model_folder) if model_folder else {}
    pinyin_config = {
        "use_g2pw": _coerce_bool(config_data.get("use_g2pw"), True),
        "dict_name": config_data.get("dict_name") or None,
    }
    model_dict["pinyin_config"] = pinyin_config

    return pinyin_config


def load_custom(model_name: str, lang: str, password="", model_cfg=None, show_info=gr.Info):
    global F5_models_dict
    model_dict = get_model_dict(model_name, lang)
    if model_dict is None:
        print("model not found")
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
            model_dict["model_folder"] = model_folder
            model_dict.pop("pinyin_config", None)
            for model_file in os.listdir(model_folder):
                if model_file.lower().endswith(".safetensors") or model_file.lower().endswith(".pt"):
                    model_dict["model"] = model_folder + "/" + model_file
                    ckpt_path = model_dict["model"]
                elif model_file.lower() == "vocab.txt":
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

    def _load_fn():
        return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

    model = _model_cache.acquire(ckpt_path, _load_fn)
    return model, ckpt_path


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
                    elif file.lower() == "vocab.txt":
                        vocab_path = model_folder + "/" + file

            model_dict = {}
            model_dict["model_name"] = model_name
            model_dict["model"] = model_path
            model_dict["vocab"] = vocab_path
            model_dict["model_folder"] = model_folder
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
                        model_dict["model_folder"] = model_folder
                        model_dict["model_url"] = ""
                        model_dict["lang"] = language
                        if language in models_dict:
                            models_dict[language].append(model_dict)
                        else:
                            models_dict[language] = [model_dict]

                        break
    print(models_dict)
    return models_dict

def load_refs_list():
    refs_root_path = "refs"
    refs_dict = {}
    speaker_dict = {}
    if not os.path.exists(refs_root_path):
        # 如果refs文件夹不存在，那就到网盘下载一份，这里需要实现网盘下载refs文件夹的功能。TODO
        return refs_dict

    for folder in os.listdir(refs_root_path):
        folder_path = os.path.join(refs_root_path, folder)
        if os.path.isdir(folder_path):
            language = folder
            ref_audio_dict = {}
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for file in filenames:
                    if file.lower().endswith(".wav") or file.lower().endswith(".mp3"):
                        ref_audio_path = os.path.join(dirpath, file)
                        ref_txt_path = ref_audio_path[:-4] + ".txt"
                        if os.path.exists(ref_txt_path):
                            key = ref_audio_path.replace("\\", "/")
                            ref_audio_dict[key] = ref_txt_path.replace("\\", "/")
                            if len(dirpath.split(os.sep)) == 3:  # 为speaker制定了单独的参考
                                speaker = dirpath.split(os.sep)[-1]
                                if speaker not in speaker_dict:
                                    speaker_dict[speaker] = []
                                if key not in speaker_dict[speaker]:
                                    speaker_dict[speaker].append(key)
            refs_dict[language] = ref_audio_dict
    return refs_dict, speaker_dict


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
refs_dict, speaker_dict = load_refs_list()
launch_in_service = True

# 推理并发数：从命令行 --concurrency 读取，默认为 1
# 注：必须在 UI 构建前解析，所以用 sys.argv 直接预读而不用 click
def _pre_parse_concurrency() -> int:
    for i, arg in enumerate(sys.argv):
        if arg in ("--concurrency", "-C") and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                pass
    return 1

infer_concurrency_limit = _pre_parse_concurrency()
print(f"[并发控制] 推理最大并发数: {infer_concurrency_limit}")

# GPU 模型缓存：最多同时驻留 infer_concurrency_limit 个模型
_model_cache = _ModelCache(infer_concurrency_limit)


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
        auto_pause,
        pause_rules,
        seed,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1,
        volume=1,
        show_info=print,
        save_line_audio=False,
        insert_punct_in_space=False,
        no_ref_audio=False,
        cfg_strength=2.0,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), [], False

    # Set inference seed
    if seed < 0 or seed > 2 ** 31 - 1:
        gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
        seed = np.random.randint(0, 2 ** 31 - 1)
    used_seed = seed
    rng_device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = torch.Generator(device=rng_device)
    rng.manual_seed(int(seed))

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    ref_audio_data = torchaudio.load(ref_audio)

    lang_alone = language.split("-", 1)[0] if "-" in language else language

    try:
        num_input = int(num_input)
        if num_input <= 0 or num_input > 20:
            gr.Warning("Invalid input box count")
            return gr.update(), [], used_seed
    except ValueError:
        gr.Warning("Input box count must be a number")
        return gr.update(), [], used_seed

    all_gen_text_list = []
    text_box_text_list = []
    for i in range(num_input):
        gen_text_box = gen_texts[i]
        if gen_text_box.strip() == "":
            continue
        index = 0
        for gen_text in gen_text_box.split("\n"):
            if gen_text.strip() == "":
                continue
            if (lang_alone == "??" and insert_punct_in_space) or "??-sit-?4" in model_name:
                gen_text = insert_punct(gen_text)

            all_gen_text_list.append(gen_text)
            index += 1

        text_box_text_list.append(index)

    if len(text_box_text_list) == 0:
        gr.Warning("No text to generate")
        return gr.update(), [], used_seed

    show_info("Loading model...")
    _load_result = load_custom(model_name, language, password)
    if _load_result is None or _load_result[0] is None:
        gr.Warning("Failed to load model")
        return gr.update(), [], used_seed
    ema_model, _infer_ckpt_path = _load_result
    pinyin_config = get_model_pinyin_config(model_name, language)

    try:
        if launch_in_service:
            os.makedirs("./gen_audio", exist_ok=True)
            gen_audio_path = tempfile.mkdtemp(dir="./gen_audio")
            delete_old_files_and_dirs("./gen_audio", days=2)

            os.makedirs("./last_audio", exist_ok=True)
            last_audio_path = tempfile.mkdtemp(dir="./last_audio")
            delete_old_files_and_dirs("./last_audio", days=2)

            os.makedirs("./tmp", exist_ok=True)
            tmp_audio_path = tempfile.mkdtemp(dir="./tmp")
            delete_old_files_and_dirs("./tmp", days=2)
        else:
            gen_audio_path = "./gen_audio"
            os.makedirs(gen_audio_path, exist_ok=True)
            last_audio_path = "./last_audio"
            os.makedirs(last_audio_path, exist_ok=True)
            tmp_audio_path = "./tmp"
            os.makedirs(tmp_audio_path, exist_ok=True)
            for file in os.listdir(tmp_audio_path):
                try:
                    if file.endswith(".wav"):
                        os.remove(os.path.join(tmp_audio_path, file))
                except OSError:
                    pass

        generated_waves = []
        progress = gr.Progress()
        start_pos = 0
        cur_box_index = 1
        total_box_count = text_box_text_list[0]
        segm_audio_list = []

        for i, gen_text in enumerate(progress.tqdm(all_gen_text_list, desc="Processing")):
            with torch.no_grad():
                final_wave, final_sample_rate, _ = infer_process(
                    ref_audio_data,
                    ref_text.lower(),
                    gen_text.lower(),
                    ema_model,
                    vocoder,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    speed=speed,
                    show_info=show_info,
                    no_ref_audio=no_ref_audio,
                    cfg_strength=cfg_strength,
                    pinyin_dict_path=pinyin_config,
                    rng=rng,
                )

            if isinstance(final_wave, torch.Tensor):
                final_wave = final_wave.cpu().numpy()
            final_wave = librosa.resample(final_wave, orig_sr=final_sample_rate, target_sr=48000)
            final_sample_rate = 48000
            generated_waves.append(final_wave)

            if save_line_audio:
                audio_filepath = os.path.join(gen_audio_path, f"{model_name}_segm_audio_{i + 1}.wav")
                sf.write(audio_filepath, final_wave, final_sample_rate, "PCM_32")
                segm_audio_list.append(audio_filepath)
            elif i == total_box_count - 1:
                final_waves = get_final_wave(cross_fade_duration, generated_waves[start_pos:], final_sample_rate)
                audio_filepath = os.path.join(gen_audio_path, f"{model_name}_segm_audio_{cur_box_index}.wav")
                sf.write(audio_filepath, final_waves, final_sample_rate, "PCM_32")
                segm_audio_list.append(audio_filepath)

                if cur_box_index < len(text_box_text_list):
                    start_pos = total_box_count
                    total_box_count += text_box_text_list[cur_box_index]
                    cur_box_index += 1

        output_audio_list = []
        ref_basename = os.path.basename(ref_audio_orig).rpartition(".")[0]
        last_gen_audio_path = os.path.join(last_audio_path, f"{model_name}--{ref_basename}--spd{speed}-orgi_audio.wav")
        final_waves = None
        if len(generated_waves) > 0:
            final_waves = get_final_wave(cross_fade_duration, generated_waves, final_sample_rate)
            sf.write(last_gen_audio_path, final_waves, final_sample_rate, "PCM_32")
            output_audio_list.append(last_gen_audio_path)

        if remove_silence and os.path.exists(last_gen_audio_path):
            remove_silence_for_generated_wav(last_gen_audio_path)
            final_waves, _ = torchaudio.load(last_gen_audio_path)
            final_waves = final_waves.squeeze().cpu().numpy()
        if volume != 1.0 and final_waves is not None:
            final_waves = final_waves * volume

        return (final_sample_rate, final_waves), output_audio_list, used_seed
    finally:
        _model_cache.release(_infer_ckpt_path)


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


def get_pinyin(lang, model_name, *input_texts):
    pinyin_config = get_model_pinyin_config(model_name, lang)
    return convert_char_to_pinyin([input_texts[0]], pinyin_config)[0]


MAX_REF_AUDIO_DURATION = 15  # 参考音频最大时长（秒）


def transcribe_with_duration_check(audio_path):
    """转录前先检测音频时长，超过限制直接返回提示，避免长音频占用转录队列"""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        info = sf.info(audio_path)
        if info.duration > MAX_REF_AUDIO_DURATION:
            gr.Warning(f"参考音频过长（{info.duration:.1f}s），请上传 {MAX_REF_AUDIO_DURATION}s 以内的音频")
            return "你输入的音频过长"
    except Exception as e:
        print(f"读取音频时长失败: {e}")
        return "读取音频时长失败"
    return asr_sherpaonnx.transcribe(audio_path)

css = """
.small-audio {
    min-height: 60px !important;
}

.small-audio audio {
    height: 32px !important;
}
"""

with gr.Blocks(title="TT-SVC_v3", css=css, analytics_enabled=False) as app:
    #     gr.Markdown(
    #         """
    # # 自定义 F5 TTS
    # """
    #     )

    def get_default_params(lang):
        global F5_models_dict
        global refs_dict
        global speaker_dict
        model_dict_list = F5_models_dict.get(lang, [])
        model_names = []
        for model_dict in model_dict_list:
            model_names.append(model_dict["model_name"])
        def_model = ""
        if len(model_names) > 0:
            model_names.reverse()
            def_model = model_names[0]

        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        ref_audios = []
        ref_txts = []

        if def_model in speaker_dict:
            ref_audios = speaker_dict[def_model]
            ref_txts = [ref_audio_dict[item] for item in ref_audios]
        else:
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

        return (model_names, def_model), def_txt, (ref_audios, def_audio), ([], None), ([], None)


    def language_change(lang):
        (model_names, def_model), def_txt, (ref_audios, def_audio), (svc_type_list, def_svc_type), (speaker_models,
                                                                                                    speaker_model) = get_default_params(
            lang)

        return gr.update(choices=model_names, value=def_model), gr.update(value=def_txt), \
            gr.update(choices=ref_audios, value=def_audio), \
            gr.update(choices=svc_type_list, value=def_svc_type), \
            gr.update(choices=speaker_models, value=speaker_model), \
            gr.update(interactive=(def_svc_type == "Applio" or def_svc_type == "RVC"))


    def get_speed(model):
        speed = 0.8
        if "-s-" in model or model.startswith("s-"):
            speed = 0.6
        elif "-d-" in model or model.startswith("d-"):
            speed = 0.9
        return speed


    def model_change(lang, model_name, ref_audio_user, orig_ref_text):
        ref_audios = []
        ref_txts = []
        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]
        ref_audio_dict = refs_dict.get(lang_alone, {})

        if model_name in speaker_dict:
            ref_audios = speaker_dict[model_name]
            ref_txts = [ref_audio_dict[item] for item in ref_audios]
        else:
            for key, value in ref_audio_dict.items():
                ref_audios.append(key)
                ref_txts.append(value)

        if len(ref_audios) > 0 and os.path.exists(ref_audios[0]):
            def_audio = ref_audios[0]
        else:
            def_audio = None

        if ref_audio_user:
            def_txt = orig_ref_text
        else:
            if len(ref_txts) > 0:
                def_txt = load_ref_txt(ref_txts[0]).strip()
            else:
                def_txt = ""

        return gr.update(value=def_txt), \
            gr.update(choices=ref_audios, value=def_audio), get_speed(model_name)


    def ref_audio_change(lang, audio_path, ref_audio_user, orig_ref_text):
        global refs_dict
        global def_txt
        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        def_txt = ""
        def_audio = ""
        for key, value in ref_audio_dict.items():
            if key == audio_path:
                def_audio = key
                def_txt = load_ref_txt(value).strip()
                break
        if ref_audio_user:
            def_txt = orig_ref_text

        return gr.update(value=def_audio), gr.update(value=def_txt)


    def svc_type_change(lang, svc_type):
        return gr.update(choices=[], value=None), gr.update(interactive=False)


    # model_manager = model_manager.WhisperModelManager()
    #
    # def transcribe_audio(audio):
    #     return asr_sherpaonnx.transcribe(audio)

    def clear_audio():
        return def_txt


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

    # with gr.Tabs():
    #     with gr.TabItem("TT"):
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
            choices=ref_audios, value=def_audio, label="参考音频", allow_custom_value=False
        )

    with gr.Row(visible=False):
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
            interactive=False,
            scale=2
        )

    # 动态布局区域
    rows = []
    max_per_row = 5
    textboxes = []

    # 创建一个动态布局，最多 20 个输入框
    for i in range(1):  # 每行最多 5 个，4 行总共 20 个
        with gr.Row(equal_height=True) as row:
            for j in range(max_per_row):
                index = i * max_per_row + j
                if index == 0:
                    textbox = gr.Textbox(label=f"生成文本:{index + 1}", lines=5, visible=True)
                    with gr.Column(scale=1):
                        pinyin_textbox = gr.Textbox(label="拼音", lines=3)
                        pinyin_button = gr.Button("查看拼音")
                    audio_output = gr.Audio(
                        label="合成音频",
                        interactive=True,
                        show_download_button=True,
                        autoplay=True,
                        sources=[],  # 不显示上传/录音入口，保留波形剪辑
                        waveform_options={"sample_rate": 48000},
                        min_width=200,
                        scale=2,
                    )
                    download_output = gr.File(label="下载文件", file_count="multiple")
                else:
                    textbox = gr.Textbox(label=f"生成文本:{index + 1}", lines=5, visible=False)
                textboxes.append(textbox)
            rows.append(row)
    pinyin_button.click(get_pinyin, inputs=[language, custom_ckpt_path, *textboxes], outputs=pinyin_textbox)
    num_input.change(create_textboxes, inputs=[num_input], outputs=textboxes)

    with gr.Row():
        clear_box_btn = gr.Button("清空文本框", variant="primary", scale=0.2, visible=False)
        generate_btn = gr.Button("合成", variant="primary")
        download_all = gr.Button("下载音频", variant="primary")

    # with gr.Accordion("高级设置", open=False):
    # gr.Markdown("✌️如果用户上传了参考音频，将会使用上传的参考，请确保参考文本和音频是一致的")
    with gr.Row(equal_height=True):
        basic_ref_audio_preset = gr.Audio(label="预设参考音频", type="filepath", value=def_audio)
        basic_ref_audio_user = gr.Audio(label="用户上传参考音频", sources=["upload"], type="filepath")
        with gr.Column():
            basic_ref_text_input = gr.Textbox(
                label="参考音频对应文本",
                lines=2,
                value=def_txt,
            )
            cb_no_ref = gr.Checkbox(label="禁用音频参考（使用时速率建议为1.0）", value=False)

        basic_ref_audio_user.upload(
            fn=transcribe_with_duration_check,
            inputs=basic_ref_audio_user,
            outputs=basic_ref_text_input,
            concurrency_id="cpu_asr",
            concurrency_limit=8,  # CPU 转录，可并发，与 GPU 推理队列相互独立
        )

        basic_ref_audio_user.clear(
            fn=clear_audio,
            inputs=None,
            outputs=basic_ref_text_input
        )
    modify_words_input = gr.Textbox(label="改词", lines=10, visible=False)
    with gr.Row(visible=False):
        enable_svc = gr.Checkbox(
            label="启用音频转换",
            info="勾选此项，启动音频转换。",
            value=False,
        )

        svc_type = gr.Dropdown(
            choices=svc_type_list,
            value=def_svc_type,
            label="音频转换类型选择",
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
        speed_slider = gr.Slider(
            label="语速设置",
            minimum=0.3,
            maximum=2.0,
            value=get_speed(def_model),
            step=0.1,
            info="调整音频语速",
        )
        volume_slider = gr.Slider(
            label="音量调节",
            minimum=0.1,
            maximum=5.0,
            value=1.0,
            step=0.1,
            info="调整音频音量",
        )
        randomize_seed = gr.Checkbox(
            label="随机种子",
            info="勾选此项，每次会自动生成随机值",
            value=True,
            scale=1,
        )
        seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
        nfe_slider = gr.Slider(
            label="运算步数",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="选择16步速度加倍，质量略降",
        )
    cfg_slider = gr.Slider(
        label="参考强度设置",
        minimum=0.0,
        maximum=10.0,
        value=2.0,
        step=0.1,
        info="值越大越像参考，值越小随机性越大",
        visible=False
    )

    cross_fade_duration_slider = gr.Slider(
        label="Cross-Fade Duration (s)",
        minimum=0.0,
        maximum=1.0,
        value=0.15,
        step=0.01,
        info="设置两个片段之前的静音时长",
        visible=False
    )
    with gr.Row(visible=True):

        remove_silence = gr.Checkbox(
            label="删除静音",
            info="模型会自动生成静音，尤其是短音频，通过此选项可以移除静音。",
            value=True,
            visible=False
        )
        save_line_audio = gr.Checkbox(
            label="按行保存音频",
            info="勾选此项，中间结果会每行保存一个音频，不勾选，则每一个文本框保存一个音频。",
            value=False,
            visible=False
        )
        insert_punct_in_space = gr.Checkbox(
            label="插入标点",
            info="此功能针对泰语，在空格处插入逗号",
            value=False,
            visible=False
        )
    with gr.Row(visible=False):
        cb_auto_pause = gr.Checkbox(
            label="标点自动间距",
            value=False
        )
        number_comma = gr.Number(label='逗号停顿（秒）', value=1.0, step=0.5)
        number_period = gr.Number(label='句号停顿（秒）', value=2.0, step=0.5)
        number_question = gr.Number(label='问号停顿（秒）', value=3.0, step=0.5)
        number_exclamation = gr.Number(label='感叹号停顿（秒）', value=3.0, step=0.5)
        number_semicolon = gr.Number(label='分号停顿（秒）', value=3.0, step=0.5)
        cb_auto_pause.change(None, cb_auto_pause, None, js="(v)=>{ setStorage('auto_pause',v) }")
        number_comma.change(None, number_comma, None, js="(v)=>{ setStorage('comma_pause',v) }")
        number_period.change(None, number_period, None, js="(v)=>{ setStorage('period_pause',v) }")
        number_question.change(None, number_question, None, js="(v)=>{ setStorage('question_pause',v) }")
        number_exclamation.change(None, number_exclamation, None,
                                  js="(v)=>{ setStorage('exclamation_pause',v) }")
        number_semicolon.change(None, number_semicolon, None, js="(v)=>{ setStorage('semicolon_pause',v) }")

    language.change(
        language_change,
        inputs=[language],
        outputs=[custom_ckpt_path, basic_ref_text_input, ref_audio, svc_type, svc_model, rvc_index_rate_slider],
        show_progress="hidden",
    )

    custom_ckpt_path.change(model_change,
                            inputs=[language, custom_ckpt_path, basic_ref_audio_user, basic_ref_text_input],
                            outputs=[basic_ref_text_input, ref_audio, speed_slider])

    ref_audio.change(
        ref_audio_change,
        inputs=[language, ref_audio, basic_ref_audio_user, basic_ref_text_input],
        outputs=[basic_ref_audio_preset, basic_ref_text_input],
        show_progress="hidden",
    )
    svc_type.change(
        svc_type_change,
        inputs=[language, svc_type],
        outputs=[svc_model, rvc_index_rate_slider],
        show_progress="hidden",
    )


    def basic_tts(
            ref_audio_preset,
            ref_audio_user,
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
            volume_slider,
            num_input,
            modify_words,
            auto_pause,
            num_comma,
            num_period,
            num_question,
            num_exclamation,
            num_semicolon,
            no_ref_audio,
            cfg_strength,
            *gen_texts_input,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2 ** 31 - 1)

        if modify_words and len(modify_words) > 0:
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
        if ref_audio_user:
            ref_audio_input = ref_audio_user
        else:
            ref_audio_input = ref_audio_preset
        audio_out, gen_audio_list, used_seed = infer(
            ref_audio_input,
            ref_text_input,
            num_input,
            gen_texts_input_modify,
            language,
            model_name,
            password,
            remove_silence,
            auto_pause,
            pause_rules={
                '，': float(num_comma),
                '。': float(num_period),
                '？': float(num_question),
                '！': float(num_exclamation),
                '；': float(num_semicolon)
            },
            seed=seed_input,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
            volume=volume_slider,
            save_line_audio=save_line_audio,
            insert_punct_in_space=insert_punct_in_space,
            no_ref_audio=no_ref_audio,
            cfg_strength=cfg_strength
        )
        return audio_out, gen_audio_list, used_seed


    intputs = [basic_ref_audio_preset,
               basic_ref_audio_user,
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
               volume_slider,
               num_input,
               modify_words_input,
               cb_auto_pause,
               number_comma,
               number_period,
               number_question,
               number_exclamation,
               number_semicolon,
               cb_no_ref,
               cfg_slider
               ] + textboxes

    generate_btn.click(
        basic_tts,
        inputs=intputs,
        outputs=[audio_output, download_output, seed_input],
        concurrency_id="gpu_infer",
        concurrency_limit=infer_concurrency_limit,
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
                                           const auto_pause = getStorage('auto_pause')
                                           const comma_pause = getStorage('comma_pause') || '1.0'
                                           const period_pause = getStorage('period_pause') || '2.0'
                                           const question_pause = getStorage('question_pause') || '3.0'
                                           const exclamation_pause = getStorage('exclamation_pause') || '3.0'
                                           const semicolon_pause = getStorage('semicolon_pause') || '3.0'
                                           return [modifyWords, auto_pause, comma_pause, period_pause, question_pause, exclamation_pause, semicolon_pause];
                                          }
                                        """
    app.load(
        None,
        inputs=None,
        outputs=[modify_words_input, cb_auto_pause, number_comma, number_period, number_question,
                 number_exclamation, number_semicolon],
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
    is_flag=True,
    default=True,
    help="Launch in service",
)
@click.option(
    "--concurrency",
    "-C",
    default=1,
    type=int,
    help="Max concurrent GPU inferences (default: 1). Set to VRAM_GB // 4 as a guideline, e.g. 2 for 8 GB.",
)
def main(port, host, share, api, root_path, inbrowser, inservice, concurrency):
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
