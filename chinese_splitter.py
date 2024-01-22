# TODO: Lingua looks like the best language detector out there for Python. But it doesn't work on code switching sentences, it needs more than a few words to detect a language switch. Maybe it can be tweaked to be more sensitive? This will allow us to work on any language, not just Chinese and English.

from enum import Enum
Lang = Enum('Lang', ('Ambiguous', 'en', 'zh'))

def get_character_type(character):
    if character >= 'a' and character <= 'z' or character >= 'A' and character <= 'Z':
        return Lang.en
    elif character >= u'\u4e00' and character <= u'\u9fa5':
        return Lang.zh
    else:
        return Lang.Ambiguous

def split_languages(sentence):
    result = []
    last_character_type = Lang.Ambiguous
    start_position = 0
    for index, character in enumerate(sentence):
        character_type = get_character_type(character)
        if character_type != last_character_type:
            if last_character_type != Lang.Ambiguous and character_type != Lang.Ambiguous:
                result.append((last_character_type, sentence[start_position:index]))
                start_position = index
        if character_type != Lang.Ambiguous:
            last_character_type = character_type
    result.append((last_character_type, sentence[start_position:]))
    return result

# sentences = [
#     "This is an English and \"你好吗。 我没事。\" mixed sentence.",
#     "2024，“待爆”还是一个好词吗？",
#     '财经| 以强大主权货币为基石 建设中国特色金融体系',
#     '券商密集发债！1月已发行550亿元，还有600亿在路上',
#     '股票| 年内14家公司股东或高管涉违规减持被罚',
#     'ETF专区 | ETF总规模上周增超 再度逼近2万亿元关口',
#     '地产空降兵 孙宏斌“减重”',
#     '科技 | “ChatGPT之父”关于人工智能有哪些新观点？',
#     'OpenAI正与全球投资者洽谈，开始筹划自己制造芯片a',
# ]

# for sentence in sentences:
#     split = split_sentence(sentence)
#     print(f'count: {len(split)}, split: {split}')