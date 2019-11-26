from collections import defaultdict
from copy import copy
from itertools import chain


# TODO imports
from scraper.lp.utils.build_aot_inv_idx import lemmatize_clean, lemmatize_twolevel as lemmatize_2l
from cluster.clusterizer import config


def split_paragraphs(text, sep='\n\n'):
    """Return paragraphs list for given text"""
    if not text:
        return []
    return filter(len, [s.strip() for s in text.split(sep)])


def read_definitions(filename):
    "Read Boris' stopwords file"

    w_groups = defaultdict(set)

    with open(filename) as fo:
        for line in fo:
            if len(line.split()) != 2:
                continue
            i, word = line.split()
            w_groups[i].add(word.decode('cp1251'))

    result = {}
    if 'L' in w_groups:
        result['body'] = w_groups['L']

    if 'H' in w_groups:
        result['title'] = w_groups['H']

    return result

def process(stop_words):
    result = {}
    for k, v in stop_words.items():
        result[k] = {'raw': v, 'lemmatized': set()}
        for word in v:
            w = list(lemmatize_clean(word))
            if len(w) == 1:
                result[k]['lemmatized'].add(w[0].decode('cp1251'))
    return result


stopwords = process(read_definitions(config.STOP_WORDS_PATH))


def helper(title, body, lemf=lemmatize_clean):
    terms = {'title': [], 'important': [], 'normal': [], 'remaining': []}
    terms['title'] = list(lemf(title, stopwords.get('title')))

    if body:
        paraghs = split_paragraphs(body)
        body_terms = list(lemf(u'\n'.join(paraghs[0:-1]),
                                          stopwords.get('body')))

        imp_len = int(len(body_terms) * config.DOC_IMPORTANT_PART_LEN)
        terms['important'] = body_terms[0:imp_len]
        terms['normal'] = body_terms[imp_len:]

        terms['remaining'] = list(lemf(paraghs[-1], stopwords.get('body')))

    # terms['title'] = list(lemf(task['title']))
    # terms['important'] = list(lemf(text[0]))
    # terms['normal'] = list(lemf('\n'.join(text[1:-1])))
    # terms['remaining'] = list(lemf(text[-1]))
    return terms

def lemmatize(title, body):
    terms = helper(title, body, lemmatize_clean)
    default = {'title': ['_EMPTY_'], 'important': [], 'normal': [], 'remaining': []}
    doclen = sum(len(t) for t in terms.values())

    return terms if doclen else copy(default)


def lemmatize_twolevel(title, body, manual_r1_terms=dict()):
    terms = {}
    lemf = lambda text, stopw: lemmatize_2l(text, stopw, type='r1')
    terms['r1'] = helper(title, body, lemf)
    lemf = lambda text, stopw: lemmatize_2l(text, stopw, type='r2')
    terms['r2'] = helper(title, body, lemf)

    all_r1_terms = set(chain.from_iterable(terms['r1'].itervalues()))
    all_r1_terms |= set(chain.from_iterable(manual_r1_terms.values()))


    for doc_part_id, doc_part in terms['r2'].items():
        terms['r2'][doc_part_id] = filter(lambda t: t not in all_r1_terms, doc_part)
        terms['r1'][doc_part_id].extend(filter(lambda t: t in all_r1_terms, doc_part))

    default = {'title': ['_EMPTY_'], 'important': [], 'normal': [], 'remaining': []}
    doclen_r1 = sum(len(t) for t in terms['r1'].values())
    doclen_r2 = sum(len(t) for t in terms['r2'].values())
    terms['r1'] = terms['r1'] if doclen_r1 else copy(default)
    terms['r2'] = terms['r2'] if doclen_r2 else copy(default)

    return terms


def similar(curr_terms, new_terms):
    assert sorted(curr_terms.keys()) == sorted(new_terms.keys())
    for k in curr_terms.keys():
        block1 = curr_terms[k]
        block2 = new_terms[k]
        if sorted(block1) != sorted(block2): return False
    return True


def similar_twolevel(curr_terms, new_terms):
    return similar(curr_terms['r1'], new_terms['r1']) and similar(curr_terms['r2'], new_terms['r2'])
