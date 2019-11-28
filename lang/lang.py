import re
import sys

from collections import defaultdict
from pathlib import Path


WORD_REGEX = re.compile(r'([\w\-\']+)', re.U | re.M)
TRIGRAM_DICT_SIZE = 500
LANG_DICT_DIR = 'data'


def words_split(text):
    for match in re.finditer(WORD_REGEX, text):
        word = match.group(0)

        yield word


def create_trigrams(text):
    trigrams = defaultdict(int)
    for word in words_split(text):
        if len(word) < 2:
            continue
        for i in range(0, len(word) - 1):
            trigram = word[i:i + 3]
            trigrams[trigram] += 1

    return trigrams


def read_dicts(dict_names):
    lang_dicts = defaultdict(lambda: defaultdict())
    for lang in dict_names:
        for line in open(f'{LANG_DICT_DIR}/{lang}.txt'):
            trigram, count = line.split(' ')
            lang_dicts[lang][trigram] = int(count)

    return lang_dicts


def compare(text_dict, lang_dict):
    intersection = 0
    union = 0
    for trigram, count in text_dict.items():
        dict_trigram_count = lang_dict.get(trigram)
        if dict_trigram_count is None:
            union += count
            continue
        intersection += dict_trigram_count
        union += dict_trigram_count

    #return intersection / (sum(text_dict.values()) + intersection)
    return intersection / union


def create_dicts(lang_dirs):
    for lang, lang_dir in lang_dirs.items():
        lang_path = Path(lang_dir)
        lang_trigrams = defaultdict(int)
        for lang_text in lang_path.iterdir():
            text = lang_text.open().read().lower()

            print(lang_trigrams)
            lang_trigrams.update(create_trigrams(text))

        lang_file = open(f'{LANG_DICT_DIR}/{lang}.txt', 'w')
        for trigram, count in sorted(lang_trigrams.items(), key=lambda x: x[-1], reverse=True)[:TRIGRAM_DICT_SIZE]:
            lang_file.write(f'{trigram} {count}\n')


def determine_lang():
    text = sys.stdin.read().lower()
    text_trigrams = create_trigrams(text)

    lang_dicts = read_dicts(['ru', 'en'])
    comparisons = defaultdict()
    for lang, lang_dict in lang_dicts.items():
        comparisons[lang] = compare(text_trigrams, lang_dict)

    filtred = list(filter(lambda x: x[-1] > 0.98, comparisons.items()))
    #print(comparisons)
    if not filtred:
        print(('none', 0.0))
    else:
        print(max(filtred, key=lambda x: x[-1]))


#create_dicts({'ru': 'source/ru', 'en': 'source/en'})
determine_lang()
