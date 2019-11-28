import re
import pymorphy2

from scraper_common.index import stopwords


morpher = pymorphy2.MorphAnalyzer()

WORD_REGEX = re.compile(r'([\w\-]+)', re.U | re.M)
BLANK_REGEX = re.compile(r'\s+', re.U | re.M)


class PostLemWord(object):
    # fake AOT plm
    def __init__(self, homonyms, start, end):
        self.homonyms = homonyms
        self.m_strWord = homonyms[0]
        self.m_strLemma = self.m_strWord.upper()
        self.m_strUpperWord = self.m_strLemma
        self.m_bFirstUpperAlpha = bool(self.m_strWord[0] == self.m_strWord[0].upper())
        self.m_GraphematicalUnitOffset = start
        self.m_TokenLengthInFile = end - start

    def GetHomonymsCount(self):
        return len(self.homonyms)

    def GetHomonym(self, i):
        return PostLemWord([self.homonyms[i]], self.m_GraphematicalUnitOffset, self.m_GraphematicalUnitOffset + self.m_TokenLengthInFile)


def words_split(text):
    for match in re.finditer(WORD_REGEX, text):
        word = match.group(0)
        if word.isdigit():
            continue

        yield word


def tokenize(text, check_stopwords=True, homonyms=True):
    for match in re.finditer(WORD_REGEX, text):
        word = match.group(0)
        if len(word) < 2 or word.isdigit():
            continue

        all_homonyms = [h.normal_form for h in morpher.parse(word)]
        if check_stopwords and any((h in stopwords.STOPWORDS for h in all_homonyms)):
            continue

        if homonyms:
            yield all_homonyms
        else:
            yield [all_homonyms[0]]


def morph(text, check_stopwords=True):
    plm_words = []
    for match in re.finditer(WORD_REGEX, text):
        word = match.group(0)
        if len(word) < 2 or word.isdigit():
            continue

        all_homonyms = [h.normal_form for h in morpher.parse(word)]
        if check_stopwords and any((h in stopwords.STOPWORDS for h in all_homonyms)):
            continue

        plm_words.append(PostLemWord(all_homonyms, match.start(), match.end()))

    return plm_words

