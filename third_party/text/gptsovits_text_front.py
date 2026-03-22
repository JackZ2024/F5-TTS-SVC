import os
import sys
import logging

import jieba_fast

# 依赖包: jieba, pypinyin, cn2an, transformers, nltk, wordsegment

# 确保 GPT-SoVITS 的根目录在 PYTHONPATH 中
# sys.path.append("/path/to/GPT-SoVITS")

try:
    # 1. 核心分词与语种识别
    from third_party.text.LangSegmenter import LangSegmenter

    # 2. 中文核心模块 (G2PW, 变调, 儿化)
    from third_party.text.chinese2 import (
        g2pw, tone_modifier, psg, is_g2pw,
        to_initials, to_finals_tone3, Style,
        correct_pronunciation, _merge_erhua,
        text_normalize as zh_normalize
    )

    # 3. 英文规范化 (数字/缩写展开)
    import third_party.text.english as sovits_en

    # 4. 符号定义
    from third_party.text.symbols import punctuation

except ImportError as e:
    print("【严重错误】无法加载 GPT-SoVITS 模块！请确保你在 GPT-SoVITS 目录下运行，或已设置 PYTHONPATH。")
    raise e

def convert_char_to_pinyin_sovits_f5(text_list, polyphone=True, f5_vocab=None):
    """
    [最终版] GPT-SoVITS 前端 -> F5-TTS 格式转换器

    特性:
    - 完整复现 inference_webui.py 的文本预处理流程
    - 支持 LangSegmenter 多语种混合切分
    - 支持 ￥/^ 等特殊停顿符号
    - 支持 G2PW 语义多音字 & 变调 & 儿化音
    - 支持 英文数字/缩写自动展开 (e.g. "No.1" -> "Number one")
    - 输出严格符合 F5-TTS Tokenizer 格式 (轻声无标号, 英文拆字符, 汉字前加空格)
    """
    final_text_list = []

    # 特殊停顿符号映射 (对应 GPT-SoVITS cleaner.py)
    # 需确保 F5 词表中有这些 Token，否则可改回 "," 或 "..."
    SPECIAL_MAP = {
        "￥": ",",
        "^": ","
    }

    def is_chinese_char(c):
        return "\u3100" <= c <= "\u9fff"

    def combine_pinyin_f5(c, v):
        """组合声韵母，适配 F5 格式 (无轻声5)"""
        v_without_tone = v[:-1]
        tone = v[-1]

        # 书写规范修正 (uie->ui, etc.)
        v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
        if c:
            # 1. 针对 j, q, x, y 的 ü -> u 还原
            if c in {'j', 'q', 'x', 'y'}:
                if v_without_tone.startswith('v'):
                    v_without_tone = 'u' + v_without_tone[1:]  # xvan -> xuan, yvan -> yuan

            # 2. 针对零声母 (y, w) 的特殊拼写处理
            if c in {'y', 'w'}:
                # y/w 开头不进行 ui, iu, un 缩写
                pyn = c + v_without_tone
            else:
                # 3. 真正的辅音声母 (b, p, m... d, t, n, l) -> 执行缩写
                # 只有辅音声母后，uei才缩写为ui，iou缩写为iu，uen缩写为un
                if v_without_tone == 'uei': v_without_tone = 'ui'
                if v_without_tone == 'iou': v_without_tone = 'iu'
                if v_without_tone == 'uen': v_without_tone = 'un'
                pyn = c + v_without_tone
        else:
            pyn_base = v_without_tone
            if pyn_base in {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}:
                pyn = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}[pyn_base]
            else:
                single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                if pyn_base and pyn_base[0] in single_rep_map:
                    pyn = single_rep_map[pyn_base[0]] + pyn_base[1:]
                else:
                    pyn = pyn_base

        return pyn if tone == "5" else pyn + tone

    def process_zh_block(text, buffer, special_token=None):
        """处理中文块：规范化 -> G2PW -> 变调 -> 儿化"""
        # 1. 中文规范化 (123 -> 一百二十三)
        text = zh_normalize(text)

        # 2. 分词准备
        seg_cut = psg.lcut(text)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)

        # 3. G2PW 预测
        if is_g2pw and polyphone:
            raw_pinyins = g2pw.lazy_pinyin(text, neutral_tone_with_five=True, style=Style.TONE3)
            print("g2pw:", raw_pinyins)
        else:
            from pypinyin import lazy_pinyin as pyp_lazy
            raw_pinyins = pyp_lazy(text, style=Style.TONE3, neutral_tone_with_five=True)

        ptr = 0
        for word, pos in seg_cut:
            print("word:", word+":"+pos)
            word_len = len(word)
            current_pinyins = raw_pinyins[ptr: ptr + word_len]
            ptr += word_len

            # 兼容逻辑：如果分词分出了纯英文/符号 (jieba bug或特殊情况)
            if not any(is_chinese_char(c) for c in word):
                if buffer and word_len > 1 and buffer[-1] != " ":
                    buffer.append(" ")
                buffer.extend(list(word))
                continue
            # --- 核心 G2P 链路 ---
            current_pinyins = correct_pronunciation(word, current_pinyins)
            sub_initials = [to_initials(p) if p and p[0].isalpha() else p for p in current_pinyins]
            sub_finals = [to_finals_tone3(p, neutral_tone_with_five=True) if p and p[0].isalpha() else p for p in
                          current_pinyins]
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)

            for i, char in enumerate(word):
                if is_chinese_char(char):
                    buffer.append(" ")  # F5 格式：汉字前空格

                    # 特殊符号处理 (如 ￥ -> , -> SP2)
                    if special_token and (sub_initials[i] in punctuation or char in "，,"):
                        pyn = special_token
                    else:
                        pyn = combine_pinyin_f5(sub_initials[i], sub_finals[i])

                    # 词表防御 (F5-Vocab Check)
                    if f5_vocab and pyn not in f5_vocab:
                        pyn = pyn.replace('r', '')  # 降级儿化音

                    buffer.append(pyn)
                else:
                    buffer.append(char)

    def process_en_block(text, buffer):
        """处理英文块：规范化 (Number Expand) -> 拆字符"""
        # 1. 英文规范化 (No.1 -> Number one) - 这是 GPT-SoVITS 区别于 F5 原版的重要细节
        text = sovits_en.text_normalize(text)
        print("text:", text)
        # 2. 简单拆分与空格处理
        # GPT-SoVITS normalize 后可能是 "Number one"，按空格切分单词
        words = text.split(" ")
        for i, word in enumerate(words):
            if not word: continue
            # 如果 buffer 非空且最后一个不是空格，则英文单词前加空格
            if buffer and buffer[-1] != " ":
                buffer.append(" ")
            buffer.extend(list(word))

    # ====== 主处理循环 ======
    for raw_text in text_list:
        char_list = []

        # [逻辑分支 A] 特殊符号检测 (cleaner.py: clean_text 的优先拦截逻辑)
        # 如果包含 ￥/^，则整句视为中文特殊模式，不再进行多语种切分
        special_hit = False
        for symbol, token in SPECIAL_MAP.items():
            if symbol in raw_text:
                # 替换为逗号进入中文处理流程
                processed_text = raw_text.replace(symbol, ",")
                process_zh_block(processed_text, char_list, special_token=token)
                special_hit = True
                break

        if special_hit:
            final_text_list.append(char_list)
            continue

        # [逻辑分支 B] 常规多语种切分模式 (inference_webui.py 核心逻辑)
        segments = LangSegmenter.getTexts(raw_text)
        print("segments:", segments)
        for seg in segments:
            lang = seg['lang']
            content = seg['text']
            if not content: continue
            # 有的中文句子被识别为日文
            if lang in ['ja', 'ko', 'zh']:
                process_zh_block(content, char_list)
            elif lang == 'en':
                process_en_block(content, char_list)
            else:  # Fallback
                process_en_block(content, char_list)  # 未知语种当英文处理或直接拆字

        final_text_list.append(char_list)

    return final_text_list
