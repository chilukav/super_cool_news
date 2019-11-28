import itertools
import math
# from datetime import datetime
from collections import Counter, defaultdict
from operator import itemgetter

# from django.conf import settings

from numpy import array
from numpy.linalg import norm
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse import coo_matrix

#from scraper import utils

from clusterizer import config
from clusterizer import common
from clusterizer.cluster import Doc
from clusterizer.cluster import MacroClusters
from clusterizer.utils.matrix import add_replace_rows
from clusterizer import log

# TODO resolve this imports. Do we actually need this shit?
# from scraper.redis import ItemDate
#from scraper.redis import TokenizedItems
#from scraper_common import db
#from scraper_common.db import models
#from scraper_common.db import lazyload


logger = log.get_logger("docspace")


def calc_tf(term_freq, doc_length, mean_length):
    tf = term_freq / (term_freq + config.BM25_K1 + config.BM25_K2 * doc_length / mean_length)
    return tf


def calc_idf(term_doc_freq):
    idf = 1.0 - config.BM25_GAMMA * max(config.BM25_RO, math.log(term_doc_freq, 10))
    return idf


def bm25(term_freq, term_doc_freq, doc_length, mean_length):
    tf_value = calc_tf(term_freq, doc_length, mean_length)
    idf_value = calc_idf(term_doc_freq)
    return tf_value * idf_value


def load_calc_idf(filename):
    import marshal

    with open(filename, 'rb') as input:
        data = marshal.load(input)

    return data


doc_mean_lens, idf_bm25 = load_calc_idf(config.BM25_IDF_PATH)


def remove_docs_twolevel(clusters, docids):
    """
    Remove docs and clusters - but doesn't recompute updated centroids.
    """
    docids = [d for d in docids if d in clusters.doc_cluster]
    if not docids:
        return dict(), []

    macro_clids = [clusters.doc_cluster[d][0] for d in docids]
    clusters.preload(macro_clids)

    docs_by_macro = defaultdict(list)
    macro_updated = defaultdict(set)
    removed_micro_clids = defaultdict(list)
    removed_macro_clids = []
    next_remove_doc_ids = []
    for d in docids:
        macro_id, micro_id = clusters.doc_cluster[d]
        del clusters.doc_cluster[d]

        doc_index = clusters[macro_id][micro_id].index(d)
        doc = clusters[macro_id][micro_id][doc_index]
        logger.debug('[remove_docs_twolevel] remove doc %s', doc)

        clusters[macro_id][micro_id].remove(d)

        docs_by_macro[macro_id].append(d)
        macro_updated[macro_id].add(micro_id)

        if not clusters[macro_id][micro_id]:
            del clusters[macro_id][micro_id]
            macro_updated[macro_id].discard(micro_id)
            removed_micro_clids[macro_id].append(micro_id)

            if not clusters[macro_id]:
                del clusters[macro_id]
                removed_macro_clids.append(macro_id)
                del macro_updated[macro_id]
                del docs_by_macro[macro_id]
                del removed_micro_clids[macro_id]
        elif doc.main_resource:
            for doc in clusters[macro_id][micro_id]:
                next_remove_doc_ids.append(doc.id)

    if removed_macro_clids:
        if clusters.keys():
            common.update_centroids_r1(clusters, [], removed_macro_clids)
        else:
            # drop everything
            MacroClusters.__init__(clusters)

    for macro_id in docs_by_macro:
        macro = clusters[macro_id]
        # clean R1 matrices
        macro.docspace_r1 = common.clean_docspace_helper(docs_by_macro[macro_id], macro.docmap_r1, macro.docspace_r1, clusters.dictionary_r1)
        # clean R2 matrices
        macro.docspace = common.clean_docspace_helper(docs_by_macro[macro_id], macro.docmap, macro.docspace, clusters.dictionary_r2)

    for macro_id, micro_clids in removed_micro_clids.items():
        clusters[macro_id].features_num = clusters.dictionary_r2.max
        common.update_centroids(clusters[macro_id], [], micro_clids)

    macro_updated = dict((macro_id, micro) for macro_id, micro in macro_updated.items() if micro)
    return macro_updated, next_remove_doc_ids


def add_doc(doc_id, terms_parts, docmap, dictionary, docspace=None):
    updated = doc_id in docmap
    doc_ndx = docmap.add(doc_id)
    term_freq = count_frequency_doc(terms_parts)
    term_freq = dict(term_freq)
    term_col = []

    for term, freq in term_freq.items():
        term_col.append((term, dictionary.add(term)))
    term_col.sort(key=itemgetter(1))
    n = docspace.shape[1] if docspace is not None else len(term_freq)
    row = csr_matrix((1, n))
    for term, col in term_col:
        if col < row.shape[1]:
            row[0, col] = term_freq[term]
    if docspace is None:
        return row

    if updated:
        docspace = add_replace_rows(docspace, [(doc_ndx, row)])
    else:
        docspace = vstack([docspace, row], format='csr')

    for term, col in term_col:
        if col >= docspace.shape[1]:
            new_column = csr_matrix((docspace.shape[0], 1))
            new_column[doc_ndx, 0] = term_freq[term]
            docspace = hstack([docspace, new_column], format='csr')
    return docspace


def count_frequency_doc(terms_parts):
    """
    Count term TF*IDF according to term position(like, title or first paragraph)
    and normalize it.
    """
    term_freq = Counter()

    doclen = {}
    doclen['title'] = len(terms_parts['title'])
    doclen['body'] = sum(len(terms) for name, terms in terms_parts.items()
                         if name != 'title')

    for part_name, terms in terms_parts.items():
        idf_name = 'title' if part_name == 'title' else 'body'
        for term, count in Counter(terms).items():
            tf = calc_tf(count, doclen[idf_name], doc_mean_lens[idf_name])
            idf = idf_bm25[idf_name].get(term, calc_idf(1))
            term_freq[term] += tf * idf * config.WEIGHTS['doc_parts'][part_name]

    term_freq = term_freq.most_common(config.DOC_VECTOR_CUT)
    lnorm = norm([freq for _, freq in term_freq])
    if lnorm:
        term_freq = [(term, freq / lnorm) for term, freq in term_freq]
    return term_freq


def preprocess_doc(terms_parts, dictionary):
    term_freq = count_frequency_doc(terms_parts)
    terms = map(itemgetter(0), term_freq)
    freqs = map(itemgetter(1), term_freq)

    row_coords = array([0] * len(term_freq))
    col_coords = array(map(dictionary.add, terms))
    data = array(freqs)
    return coo_matrix((data, (row_coords, col_coords)), shape=(1, dictionary.max))


def prepare_docs(clusters, item_ids, clean, baker_data):
    all_tokens = TokenizedItems().mget(item_ids)
    doc_terms = {doc_id: terms for doc_id, terms in zip(item_ids, all_tokens) if terms}

    items = []
    if doc_terms.keys():
        with db.session(master=True, commit=False) as session:
            items = session.query(
                models.Item.id,
                models.Item.fulltext_status,
                models.Item.resource_id,
                models.Item.mtime,
                models.Resource.rating,
                models.Resource.autorating
            ).join(models.Resource, models.Resource.id == models.Item.resource_id).filter(
                models.Item.id.in_(doc_terms.keys())
            )

    added = []
    updated = []
    for item in items:
        doc = Doc(
            id=item.id,
            fulltext=item.fulltext_status or 0,
            source_rating=item.rating or item.autorating or 0,
            mtime=item.mtime,
            resource_id=item.resource_id,
        )

        if item.id in clusters.doc_cluster:
            updated.append(doc)
        else:
            added.append(doc)

    removed_docids = []
    # if clean and baker_data['lastclean'] < (datetime.now() - settings.SCRAPER_CONSTANTS['baker']['items_cleanup_timedelta']):
    #     removed_docids = ItemDate().outdated(datetime.now() - settings.ITEM_ACTIVE_TIMEDELTA)
    #     baker_data['lastclean'] = datetime.now()
    #     removed_docids = list(set(removed_docids) - set(item_ids))
    #     log.bark('[prepare_docs] removed docids: {}'.format(removed_docids))

    next_remove_doc_ids = remove_docs_twolevel(clusters, removed_docids)[1]
    if next_remove_doc_ids:
        logger.debug('[prepare_docs] remove cascade: %s', next_remove_doc_ids)
        remove_docs_twolevel(clusters, next_remove_doc_ids)

    logger.debug('[prepare_docs] updated: %s', [doc.id for doc in updated])
    macro_updated = remove_docs_twolevel(clusters, [doc.id for doc in updated])[0]

    for doc in itertools.chain(added, updated):
        doc.vector_r1 = preprocess_doc(doc_terms[doc.id]['r1'], clusters.dictionary_r1)
        doc.vector_r2 = preprocess_doc(doc_terms[doc.id]['r2'], clusters.dictionary_r2)

    return added, updated, removed_docids, macro_updated
