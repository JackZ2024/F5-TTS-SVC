# This code is modified from https://github.com/mozillazg/pypinyin-g2pW

import ast
import pickle
import os
from importlib.resources import files

import jieba_fast
from pypinyin.constants import RE_HANS
from pypinyin.core import Pinyin, Style
from pypinyin.seg.simpleseg import simple_seg
from pypinyin.converter import UltimateConverter
from pypinyin.contrib.tone_convert import to_tone, to_tone3
from .onnx_api import G2PWOnnxConverter

current_file_path = os.path.dirname(__file__)
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")
jieba_fast_added_words = set()


class G2PWPinyin(Pinyin):
    def __init__(
            self,
            model_dir="G2PWModel/",
            model_source=None,
            enable_non_tradional_chinese=True,
            v_to_u=False,
            neutral_tone_with_five=False,
            tone_sandhi=False,
            **kwargs,
    ):
        self._g2pw = G2PWOnnxConverter(
            model_dir=model_dir,
            style="pinyin",
            model_source=model_source,
            enable_non_tradional_chinese=enable_non_tradional_chinese,
        )
        self._converter = Converter(
            self._g2pw,
            v_to_u=v_to_u,
            neutral_tone_with_five=neutral_tone_with_five,
            tone_sandhi=tone_sandhi,
        )

    def get_seg(self, **kwargs):
        return simple_seg


class Converter(UltimateConverter):
    def __init__(self, g2pw_instance, v_to_u=False, neutral_tone_with_five=False, tone_sandhi=False, **kwargs):
        super(Converter, self).__init__(
            v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi, **kwargs
        )

        self._g2pw = g2pw_instance

    def convert(self, words, style, heteronym, errors, strict, **kwargs):
        pys = []
        if RE_HANS.match(words):
            pys = self._to_pinyin(words, style=style, heteronym=heteronym, errors=errors, strict=strict)
            post_data = self.post_pinyin(words, heteronym, pys)
            if post_data is not None:
                pys = post_data

            pys = self.convert_styles(pys, words, style, heteronym, errors, strict)

        else:
            py = self.handle_nopinyin(words, style=style, errors=errors, heteronym=heteronym, strict=strict)
            if py:
                pys.extend(py)

        return _remove_dup_and_empty(pys)

    def _to_pinyin(self, han, style, heteronym, errors, strict, **kwargs):
        pinyins = []

        g2pw_pinyin = self._g2pw(han)

        if not g2pw_pinyin:  # g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
            return super(Converter, self).convert(han, Style.TONE, heteronym, errors, strict, **kwargs)

        for i, item in enumerate(g2pw_pinyin[0]):
            if item is None:  # g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
                py = super(Converter, self).convert(han[i], Style.TONE, heteronym, errors, strict, **kwargs)
                pinyins.extend(py)
            else:
                pinyins.append([to_tone(item)])

        return pinyins


def _remove_dup_items(lst, remove_empty=False):
    new_lst = []
    for item in lst:
        if remove_empty and not item:
            continue
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def _remove_dup_and_empty(lst_list):
    new_lst_list = []
    for lst in lst_list:
        lst = _remove_dup_items(lst, remove_empty=True)
        if lst:
            new_lst_list.append(lst)
        else:
            new_lst_list.append([""])

    return new_lst_list


def resolve_pinyin_dict_path(pinyin_dict_path=None):
    if pinyin_dict_path and os.path.exists(pinyin_dict_path):
        return os.path.abspath(pinyin_dict_path)

    default_path = str(files("f5_tts").joinpath("dicts/pypinyin.txt"))
    return os.path.abspath(default_path) if os.path.exists(default_path) else None


def get_dict(pinyin_dict_path=None):
    polyphonic_dict = read_dict(pinyin_dict_path)

    return polyphonic_dict


def read_dict(pinyin_dict_path=None):
    polyphonic_dict = {}
    with open(PP_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = ast.literal_eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    with open(PP_FIX_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = ast.literal_eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    pypinyin_dict_file = resolve_pinyin_dict_path(pinyin_dict_path)
    if pypinyin_dict_file and os.path.exists(pypinyin_dict_file):
        for line in open(pypinyin_dict_file, "r", encoding="utf-8").read().split("\n"):
            if "#" in line:
                continue
            parts = line.split(":")
            if len(parts) == 2:
                polyphonic_dict[parts[0].strip()] = [to_tone3(item) for item in parts[1].strip().split(" ")]
    for word in polyphonic_dict:
        if word not in jieba_fast_added_words:
            jieba_fast.add_word(word)
            jieba_fast_added_words.add(word)
    return polyphonic_dict


_pp_dict_cache = {}


def get_pp_dict(pinyin_dict_path=None):
    cache_key = resolve_pinyin_dict_path(pinyin_dict_path)
    if cache_key not in _pp_dict_cache:
        _pp_dict_cache[cache_key] = get_dict(cache_key)
    return _pp_dict_cache[cache_key]


def preload_pp_dicts():
    dicts_dir = str(files("f5_tts").joinpath("dicts"))
    if not os.path.isdir(dicts_dir):
        return 0

    count = 0
    for file_name in os.listdir(dicts_dir):
        if not file_name.startswith("g2pw"):
            continue
        file_path = os.path.join(dicts_dir, file_name)
        if os.path.isfile(file_path):
            get_pp_dict(file_path)
            count += 1
    return count


def correct_pronunciation(word, word_pinyins, pinyin_dict_path=None):
    current_pp_dict = get_pp_dict(pinyin_dict_path)
    new_pinyins = current_pp_dict.get(word, "")
    if new_pinyins == "":
        for idx, w in enumerate(word):
            w_pinyin = current_pp_dict.get(w, "")
            if w_pinyin != "":
                word_pinyins[idx] = w_pinyin[0]
        return word_pinyins
    else:
        return new_pinyins
