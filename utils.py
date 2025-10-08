from pydub import AudioSegment
from pydub.silence import detect_silence


def detect_silence_regions(audio, min_silence_len=50, silence_thresh=-40):
    """
    检测音频中的静音区域

    Args:
        audio: AudioSegment对象
        min_silence_len: 最小静音长度(ms)
        silence_thresh: 静音阈值(dB)

    Returns:
        List of tuples: [(start_ms, end_ms), ...]
    """
    silence_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    return silence_ranges


def find_nearest_silence(audio, target_ms, search_range=50, min_silence_len=30, silence_thresh=-40):
    """
    在目标位置附近寻找最近的静音区

    Args:
        audio: AudioSegment对象
        target_ms: 目标时间点(ms)
        search_range: 搜索范围(ms)，前后各search_range
        min_silence_len: 最小静音长度(ms)
        silence_thresh: 静音阈值(dB)

    Returns:
        int or None: 找到的静音区中点位置(ms)，找不到返回None
    """
    # 确保搜索范围在音频长度内
    start_search = max(0, target_ms - search_range)
    end_search = min(len(audio), target_ms + search_range)

    # 提取搜索区域的音频
    search_audio = audio[start_search:end_search]

    # 检测该区域的静音
    silence_ranges = detect_silence(
        search_audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not silence_ranges:
        return None

    # 找到距离目标位置最近的静音区中点
    best_position = None
    min_distance = float('inf')

    for silence_start, silence_end in silence_ranges:
        # 转换为绝对位置
        abs_silence_start = start_search + silence_start
        abs_silence_end = start_search + silence_end
        silence_center = (abs_silence_start + abs_silence_end) // 2

        # 计算距离目标位置的距离
        distance = abs(silence_center - target_ms)
        if distance < min_distance:
            min_distance = distance
            best_position = silence_center

    return best_position


def is_in_silence(audio, position_ms, silence_thresh=-40, window_ms=10):
    """
    检查指定位置是否在静音区

    Args:
        audio: AudioSegment对象
        position_ms: 检查的位置(ms)
        silence_thresh: 静音阈值(dB)
        window_ms: 检查窗口大小(ms)

    Returns:
        bool: 是否在静音区
    """
    start = max(0, position_ms - window_ms // 2)
    end = min(len(audio), position_ms + window_ms // 2)

    if start >= end:
        return True  # 边界情况认为是静音

    audio_segment = audio[start:end]
    return audio_segment.dBFS < silence_thresh


def auto_pause_by_whisper(model, audio_file, reference_text, pause_rules):
    """
    改进的自动停顿插入函数，会在静音区插入停顿

    Args:
        model: 对齐模型
        audio_file: 音频文件路径
        reference_text: 参考文本
        pause_rules: 标点符号停顿规则字典

    Returns:
        str: 输出文件路径
    """
    # 步骤1: 对齐音频与参考文本
    result = model.align(
        audio_file,
        reference_text,
        language='zh',
        suppress_silence=True,
        vad=True,
        min_word_dur=0.02,
        nonspeech_error=0.3,
        max_word_dur=5.0,
    )

    result = result.split_by_punctuation(pause_rules.keys()).merge_by_gap(0.1, max_words=None)

    # 步骤2: 加载音频
    audio = AudioSegment.from_file(audio_file)
    output_audio = AudioSegment.empty()
    words = result.all_words()

    print(f"处理音频长度: {len(audio)}ms, 词数: {len(words)}")

    prev_end = 0
    processed_positions = []  # 记录处理过的位置，用于调试

    for i, word in enumerate(words):
        if i == len(words) - 1:  # 最后一个词不用加间距
            break
        start_ms = int(word.start * 1000)
        # 词尾要留出一些尾息
        end_ms = int((word.end + 0.2 * (words[i + 1].start - word.end))  * 1000)

        # 添加从上一个结束位置到当前词开始的音频
        if start_ms > prev_end:
            gap_audio = audio[prev_end:start_ms]
            output_audio += gap_audio

        # 添加当前词的音频
        word_audio = audio[start_ms:end_ms]
        output_audio += word_audio
        prev_end = end_ms

        # 检查词末尾是否有需要添加停顿的标点
        text = word.word.strip()
        if not text:
            continue

        # 查找标点符号
        punctuation_found = None
        for punct in pause_rules:
            if text.endswith(punct):
                punctuation_found = punct
                break

        if punctuation_found:
            pause_duration = pause_rules[punctuation_found] * 1000  # 转换为毫秒
            target_position = end_ms  # 词结束位置

            print(f"词 '{text}' 在位置 {target_position}ms 需要添加 {pause_duration}ms 停顿")

            # 检查目标位置是否在静音区
            if is_in_silence(audio, target_position, silence_thresh=-40):
                # 直接在当前位置添加停顿
                print(f"  -> 当前位置已在静音区，直接添加停顿")
                output_audio += AudioSegment.silent(duration=pause_duration)
                processed_positions.append((target_position, target_position, "direct"))
            else:
                # 寻找附近的静音区
                silence_position = find_nearest_silence(
                    audio,
                    target_position,
                    search_range=50,
                    min_silence_len=20,
                    silence_thresh=-40
                )

                if silence_position is not None:
                    print(f"  -> 在位置 {silence_position}ms 找到静音区，调整停顿位置")

                    # 需要先添加到静音位置的音频
                    if silence_position > prev_end:
                        bridge_audio = audio[prev_end:silence_position]
                        output_audio += bridge_audio
                        prev_end = silence_position

                    # 在静音位置添加停顿
                    output_audio += AudioSegment.silent(duration=pause_duration)
                    processed_positions.append((target_position, silence_position, "adjusted"))
                else:
                    print(f"  -> 未找到合适的静音区，跳过停顿添加")
                    processed_positions.append((target_position, None, "skipped"))

    # 添加剩余的音频
    if prev_end < len(audio):
        remaining_audio = audio[prev_end:]
        output_audio += remaining_audio

    # 导出结果
    output_file = audio_file.replace(".wav", "_adjusted.wav")
    output_audio.export(output_file, format='wav')

    # 打印处理总结
    print(f"\n处理完成:")
    print(f"原始音频长度: {len(audio)}ms")
    print(f"输出音频长度: {len(output_audio)}ms")
    print(f"处理的停顿位置: {len(processed_positions)}")

    for orig_pos, final_pos, action in processed_positions:
        if action == "direct":
            print(f"  {orig_pos}ms: 直接添加停顿")
        elif action == "adjusted":
            print(f"  {orig_pos}ms -> {final_pos}ms: 调整到静音区")
        else:
            print(f"  {orig_pos}ms: 跳过（未找到静音区）")

    return output_file


# 使用示例
if __name__ == "__main__":
    # 示例标点规则（秒）
    pause_rules = {
        '，': 0.3,
        '。': 0.5,
        '？': 0.5,
        '！': 0.5,
        '；': 0.4,
        '：': 0.4,
        '、': 0.2
    }
