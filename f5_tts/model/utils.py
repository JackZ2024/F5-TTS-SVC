from __future__ import annotations

import random
from collections import defaultdict
from importlib.resources import files
import re
import os
import jieba
from pypinyin import lazy_pinyin, Style, load_phrases_dict
import torch
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
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

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
def convert_zh_mix_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    global dict_loaded
    if not dict_loaded:
        dict_loaded = True
        jieba_dict_file = str(files("f5_tts").joinpath(f"dicts/jieba.txt"))
        if os.path.exists(jieba_dict_file):
            print(f"加载jieba词典{jieba_dict_file}")
            jieba.load_userdict(jieba_dict_file)

        pypinyin_dict_file = str(files("f5_tts").joinpath(f"dicts/pypinyin.txt"))
        if os.path.exists(pypinyin_dict_file):
            print(f"加载pypinyin词典{pypinyin_dict_file}")
            load_phrases_dict(load_pypinyin_dict_file(pypinyin_dict_file))

    final_text_list = []
    custom_trans = str.maketrans(
        # {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        {"“": '', "”": '', "\"": "", "‘": "", "’": "", "'": ""}
    )  # add custom trans here, to address oov

    def has_toned_pinyin(text):
        """检测文本中是否包含带声调的拼音（声调符号或数字声调）"""
        words = text.split()
        for word in words:
            if not word:
                continue

            # 检查是否包含声调符号
            tone_chars = 'āáǎàēéěèīíǐìōóǒòūúǔùüǖǘǚǜ'
            if any(c in tone_chars for c in word):
                return True

            # 检查是否以1-4数字结尾（数字声调）
            if len(word) > 1 and word[-1] in '1234':
                return True

        return False

    def normalize_pinyin_tone(pinyin):
        """将带声调符号的拼音转换为数字声调格式"""
        tone_map = {
            'ā': 'a1', 'á': 'a2', 'ǎ': 'a3', 'à': 'a4',
            'ē': 'e1', 'é': 'e2', 'ě': 'e3', 'è': 'e4',
            'ī': 'i1', 'í': 'i2', 'ǐ': 'i3', 'ì': 'i4',
            'ō': 'o1', 'ó': 'o2', 'ǒ': 'o3', 'ò': 'o4',
            'ū': 'u1', 'ú': 'u2', 'ǔ': 'u3', 'ù': 'u4',
            'ü': 'v', 'ǖ': 'v1', 'ǘ': 'v2', 'ǚ': 'v3', 'ǜ': 'v4'
        }

        result = pinyin.lower()
        for char, replacement in tone_map.items():
            result = result.replace(char, replacement)

        return result

    def is_likely_pinyin_word(word):
        """在已确定使用拼音的情况下，判断单词是否可能是拼音"""
        if not word:
            return False

        word_lower = word.lower()

        # 去掉可能的数字声调
        if word[-1] in '1234':
            base = word_lower[:-1]
        else:
            base = word_lower

        # 简单检查：只包含字母
        return re.match(r'^[a-z]+$', base) is not None

    def process_mixed_segment(text):
        """处理包含中文和拼音的混合文本段"""
        char_list = []
        words = text.split()

        for word in words:
            if not word:
                continue

            # 检查是否包含中文字符和拉丁字符
            has_chinese = any(is_chinese(c) for c in word)
            has_latin = any(c.isalpha() and ord(c) < 256 for c in word)

            if has_chinese and has_latin:
                # 中文和拼音混在一起，需要分离处理
                i = 0
                current_pinyin = ""

                while i < len(word):
                    char = word[i]

                    if is_chinese(char):
                        # 如果当前有拼音段，先处理拼音段
                        if current_pinyin:
                            if is_likely_pinyin_word(current_pinyin):
                                if char_list and char_list[-1] != " ":
                                    char_list.append(" ")
                                char_list.append(normalize_pinyin_tone(current_pinyin))
                            else:
                                # 不是拼音，按字符添加
                                if char_list and len(current_pinyin) > 1 and char_list[-1] not in " :'\"":
                                    char_list.append(" ")
                                char_list.extend(current_pinyin)
                            current_pinyin = ""

                        # 处理中文字符
                        if char_list:
                            char_list.append(" ")
                        char_list.extend(lazy_pinyin(char, style=Style.TONE3, tone_sandhi=True))

                    elif char.isalpha() and ord(char) < 256:
                        # 拉丁字母，累积到拼音段
                        current_pinyin += char

                    elif char.isdigit() and current_pinyin:
                        # 数字声调，添加到拼音段
                        current_pinyin += char

                    else:
                        # 其他字符（标点等）
                        # 先处理累积的拼音
                        if current_pinyin:
                            if is_likely_pinyin_word(current_pinyin):
                                if char_list and char_list[-1] != " ":
                                    char_list.append(" ")
                                char_list.append(normalize_pinyin_tone(current_pinyin))
                            else:
                                if char_list and len(current_pinyin) > 1 and char_list[-1] not in " :'\"":
                                    char_list.append(" ")
                                char_list.extend(current_pinyin)
                            current_pinyin = ""

                        # 处理标点符号
                        char_list.append(char)

                    i += 1

                # 处理剩余的拼音段
                if current_pinyin:
                    if is_likely_pinyin_word(current_pinyin):
                        if char_list and char_list[-1] != " ":
                            char_list.append(" ")
                        char_list.append(normalize_pinyin_tone(current_pinyin))
                    else:
                        if char_list and len(current_pinyin) > 1 and char_list[-1] not in " :'\"":
                            char_list.append(" ")
                        char_list.extend(current_pinyin)

            elif has_chinese:
                # 纯中文，按原逻辑处理
                for seg in jieba.cut(word):
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
            elif is_likely_pinyin_word(word):
                # 纯拼音单词，直接添加
                if char_list and char_list[-1] != " ":
                    char_list.append(" ")
                char_list.append(normalize_pinyin_tone(word))
            else:
                # 其他情况（标点符号等），直接添加
                if char_list and len(word) > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(word)

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

        final_text_list.append(char_list)

    return final_text_list


def convert_char_to_pinyin(text_list, polyphone=True):
    if any(is_chinese(c) for c in text_list[0]):
        return convert_zh_mix_char_to_pinyin(text_list)

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
