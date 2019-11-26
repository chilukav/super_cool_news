from collections import defaultdict
from numpy import zeros
from numpy.linalg import norm
from scipy.sparse import csr_matrix, coo_matrix

from cluster.clusterizer.utils.matrix import add_replace_rows, set_columns_number, remove_rows
from cluster.clusterizer import config
from cluster.clusterizer import log

logger = log.get_logger("common")


def build_centroid(doc_indexes, docspace):
    v = docspace[doc_indexes, :].mean(axis=0)
    return v / norm(v)


def build_centroids(doc_buckets, docspace, docmap):
    '''Build centroids for given clusters.
    doc_buckets could be a Clusters(), or, for example, map: clid -> list of docs

    Returns: list (centroid_index -> clid), matrix of centroids.
    '''
    centroids = zeros((len(doc_buckets), docspace.shape[1]))
    clids = []
    for i, clid in enumerate(doc_buckets):
        clids.append(clid)
        doc_indexes = [docmap[doc.id] for doc in doc_buckets[clid]]
        centroids[i] = build_centroid(doc_indexes, docspace)

    return clids, csr_matrix(centroids)


def build_centroid_r1(cluster):
    v = cluster.docspace_r1.mean(axis=0)
    return coo_matrix(v / norm(v))


def update_centroids_r1(clusters, updated_clids, removed_clids):
    updated_centroids = []
    for clid in updated_clids:
        if getattr(clusters[clid], 'fixed_centroid', False):
            continue
        updated_centroids.append((clid, build_centroid_r1(clusters[clid])))

    clusters.centroids_r1 = update_docspace_helper(clusters.clmap_r1, clusters.centroids_r1,
                                updated_centroids, removed_clids, clusters.dictionary_r1.max)


def update_centroids(clusters, updated_clids, removed_clids):
    docspace, docmap = clusters.docspace, clusters.docmap

    updated_centroids = []
    for clid in updated_clids:
        doc_indexes = [docmap[id] for id in clusters[clid].doc_ids()]
        updated_centroids.append((clid, coo_matrix(build_centroid(doc_indexes, docspace))))

    clusters.centroids = update_docspace_helper(clusters.clmap, clusters.centroids,
                            updated_centroids, removed_clids, clusters.features_num)


def update_docspace_helper(docmap, matrix, updated_rows, removed_ids, features_num):
    "Add and remove rows, but doesn't change dictionary"
    if matrix is not None:
        if removed_ids:
            row_ndxes = [docmap[id] for id in removed_ids]
            matrix = remove_rows(matrix, row_ndxes)
            docmap.remove_docs(removed_ids)
            logger.debug('[update_docspace_helper] Remove docs from helper: %s', removed_ids)
        matrix = set_columns_number(matrix, features_num)
    else:
        matrix = csr_matrix((len(updated_rows), features_num))

    rows = []
    for id, centroid in updated_rows:
        idx = docmap.add(id)
        rows.append((idx, set_columns_number(centroid, features_num)))

    if rows:
        return add_replace_rows(matrix, rows)
    else:
        return matrix


def clean_docspace_helper(removed_ids, docmap, docspace, dictionary):
    "Cleans docmap docspace and dictionary"
    doc_ndxes = [docmap[id] for id in removed_ids]
    doc_ndxes.sort()

    # get columns, that are affected and can become zero
    rows, cols = docspace[doc_ndxes, :].nonzero()
    columns = defaultdict(int)
    for c in cols:
        columns[c] += 1

    # remove rows from docspace
    docspace = remove_rows(docspace, doc_ndxes)

    # clean dictionary
    for c, count in columns.items():
        term = dictionary.by_index(c)
        dictionary.remove(term, count)

    # clear docmap
    docmap.remove_docs(removed_ids)

    return docspace


def clusters_centroids(clusters, clids):
    clusters.centroids = set_columns_number(clusters.centroids, clusters.features_num)
    idxes = [clusters.clmap[clid] for clid in clids]
    return clusters.centroids[idxes]


def proximity_matrix(matrix1, matrix2):
    '''Build proximity matrix.'''
    return matrix1 * matrix2.transpose()


def complex_proximities(docs, proximities):
    # FIXME: main doc should have proximity 1 anyway, so normalize this ?

    max_time = max([d.mtime for d in docs])
    min_time = max_time - config.ITEM_ACTIVE_TIMEDELTA

    result = []
    for doc, prox in zip(docs, proximities):
        if doc.fulltext:
            res = prox + config.WEIGHTS['fulltext'] * (1 - prox)
        else:
            res = prox
        res *= (doc.source_rating * 0.01)
        res *= (doc.mtime - min_time).total_seconds() / config.ITEM_ACTIVE_TIMEDELTA.total_seconds()
        result.append(res)

    return result


def update_main_docs(clusters, clids):
    clusters.docspace = set_columns_number(clusters.docspace, clusters.features_num)

    docspace, docmap = clusters.docspace, clusters.docmap
    centroids = clusters_centroids(clusters, clids)

    proximities = proximity_matrix(centroids, docspace)

    # Find main docs
    for centroid_ndx, clid in enumerate(clids):
        if getattr(clusters[clid], 'manual_pitem', None) and clusters[clid].manual_pitem in clusters[clid]:
            main_doc_ndx = clusters[clid].index(clusters[clid].manual_pitem)
        else:
            docs_proximities = [proximities[centroid_ndx, docmap[doc.id]] for doc in clusters[clid]]
            docs_proximities = complex_proximities(clusters[clid], docs_proximities)
            main_doc_ndx = docs_proximities.index(max(docs_proximities))

        clusters[clid][main_doc_ndx].status = 'main'
        clusters[clid][main_doc_ndx].proximity = 1
        clusters[clid].insert(0, clusters[clid].pop(main_doc_ndx))
