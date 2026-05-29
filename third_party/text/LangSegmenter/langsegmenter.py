
class LangSegmenter():
    @staticmethod
    def getTexts(text, default_lang=""):
        if default_lang:
            return [{"lang": default_lang, "text": text}]

        if not text:
            return []

        # --- 核心逻辑：基于字符特征的物理切分 ---

        # 1. 定义字符属性判定函数
        def get_char_lang(char):
            # 判英文字母 (A-Z, a-z)
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                return 'en'
            # 判汉字 (CJK 统一汉字范围)
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                return 'zh'
            # 判数字
            if '0' <= char <= '9':
                return 'digit'
            # 判空白
            if char.isspace():
                return 'space'
            # 其余视为标点/特殊符号
            return 'other'

        # 2. 状态机切分
        segments = []
        if not text:
            return []

        current_lang = 'zh'  # 默认初始状态
        current_text = ""

        for char in text:
            ctype = get_char_lang(char)

            # 决策逻辑：
            # - en/zh/digit 相互触发切换
            # - space 和 other (标点) 采取 "就近原则" (粘附在前面的语块上)

            if ctype in ['zh', 'en', 'digit']:
                # 如果当前块是空的，或者当前字符类型与块类型一致，或者是数字粘附
                if not current_text:
                    current_lang = ctype
                    current_text = char
                elif ctype == current_lang:
                    current_text += char
                elif ctype == 'digit' and current_lang in ['zh', 'en']:
                    # 数字不触发切换，直接粘附在之前的语种后
                    current_text += char
                else:
                    # 触发切换
                    segments.append({'lang': current_lang, 'text': current_text})
                    current_lang = ctype
                    current_text = char
            else:
                # 标点和空格：不触发切换，直接跟着前面的跑
                if not current_text:
                    current_text = char
                else:
                    current_text += char

        if current_text:
            segments.append({'lang': current_lang, 'text': current_text})

        # 3. 结果精修
        final_list = []
        for seg in segments:
            # 规范化标记：将 digit 和 zh 统称为 zh (因为中文流程会处理数字)
            l = 'en' if seg['lang'] == 'en' else 'zh'

            # 简单的段合并
            if final_list and final_list[-1]['lang'] == l:
                final_list[-1]['text'] += seg['text']
            else:
                final_list.append({'lang': l, 'text': seg['text']})

        # 4. 后处理：处理边缘情况（例如纯标点块的语种归属）
        # 如果一段文字全是标点，默认让它跟着前面的语种。
        return final_list


if __name__ == "__main__":
    # 测试那些 AI 处理不好的案例
    test_cases = [
        "什么餐都行，AI什么都会做。",
        "这条WiFi信号不行啊。",
        "123,这是数字开头。",
        "Xi'an is a city. 西安是个城市。",
    ]
    for t in test_cases:
        res = LangSegmenter.getTexts(t)
        print(f"Input: {t}\nOutput: {res}\n")