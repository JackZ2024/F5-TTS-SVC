from __future__ import annotations

import random
from collections import defaultdict
from importlib.resources import files
import re
import os
import jieba
from pypinyin import lazy_pinyin, Style, load_phrases_dict
import torch
from pypinyin.constants import PHRASES_DICT
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
        text: list[str] | list[list[str]],
        vocab_char_map: dict[str, int],  # {char: idx}
        padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz space is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin

def load_pypinyin_dict_file(file_path):
    """
    加载自定义拼音词典文件。
    文件格式示例： 好使唤人: hào shǐ huàn rén
    转换为 pypinyin 格式： {'好使唤人': [['hào'], ['shǐ'], ['huàn'], ['rén']]}
    """
    custom_phrases = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行或无效行
            if ':' not in line:
                continue

            # 分割词语和拼音部分
            word, pinyins_str = line.split(':', 1)
            word = word.strip()

            # 按空格分割拼音字符串
            pinyins_list = pinyins_str.strip().split()

            # 将拼音列表转换为 pypinyin 需要的嵌套列表格式
            # 例如 ['hào', 'shǐ'] -> [['hào'], ['shǐ']]
            formatted_pinyins = [[p] for p in pinyins_list]

            if word:
                custom_phrases[word] = formatted_pinyins

    return custom_phrases


def is_chinese(c):
    return "\u3100" <= c <= "\u9fff"


dict_loaded = False


# 中文混合输入转拼音，汉字可以混入拼音
def convert_zh_mix_char_to_pinyin(text_list, pinyin_dict_path, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    global dict_loaded
    if not dict_loaded:
        dict_loaded = True

        if pinyin_dict_path and os.path.exists(pinyin_dict_path):    # 推理的时候放在权重文件夹指定
            pypinyin_dict_file = pinyin_dict_path
        else:                                                        # 微调的时候放在固定dicts文件夹
            pypinyin_dict_file = str(files("f5_tts").joinpath(f"dicts/pypinyin.txt"))
        if os.path.exists(pypinyin_dict_file):
            print(f"加载pypinyin词典{pypinyin_dict_file}")
            load_phrases_dict(load_pypinyin_dict_file(pypinyin_dict_file))
            for word in PHRASES_DICT:
                jieba.add_word(word)

    final_text_list = []
    custom_trans = str.maketrans(
        # {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        {"“": '', "”": '', "\"": "", "‘": "", "’": "", "'": ""}
    )  # add custom trans here, to address oov

    def has_toned_pinyin(text):
        return "<" in text and ">" in text

    def split_by_brackets(text):
        """按尖括号分割字符串"""
        result = re.split(r'<|>', text)
        # 过滤空字符串
        return [s for s in result if s]

    def convert_single(polyphone: bool, text) -> list[str]:
        char_list = []
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        return char_list

    def process_mixed_segment(text):
        """处理包含中文和拼音的混合文本段"""
        char_list = []
        parts = split_by_brackets(text)
        for part in parts:
            if any(is_chinese(c) for c in part):
                char_list.extend(convert_single(polyphone, part))
            else:
                char_list.append(" ")
                char_list.append(part)
        return char_list

    for text in text_list:
        text = text.translate(custom_trans)

        # 检查文本是否包含带声调的拼音
        has_pinyin = has_toned_pinyin(text)

        if has_pinyin:
            # 检测到带声调拼音，将所有单词都当作拼音处理
            char_list = process_mixed_segment(text)
        else:
            # 不包含拼音，使用原有逻辑
            char_list = convert_single(polyphone, text)

        final_text_list.append(char_list)

    return final_text_list


def convert_char_to_pinyin(text_list, pinyin_dict_path=None, polyphone=True):
    if any(is_chinese(c) for c in text_list[0]):
        return convert_zh_mix_char_to_pinyin(text_list, pinyin_dict_path)

    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i: i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
