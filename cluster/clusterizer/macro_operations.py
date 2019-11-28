import time
from collections import defaultdict
from datetime import datetime
from operator import attrgetter
from numpy import argmax, argwhere
from scipy.sparse import coo_matrix


from clusterizer import common
from clusterizer import config
from clusterizer.common import update_docspace_helper
from clusterizer.common import update_centroids_r1
from clusterizer.cluster import Docmap
from clusterizer.cluster import MacroCluster
from clusterizer.cluster import get_cluster_id
from clusterizer.docspace import remove_docs_twolevel
from clusterizer.utils.matrix import append_rows
from clusterizer.utils.matrix import set_columns_number
# TODO imports
#from cluster.lp.utils.build_aot_inv_idx import lemmatize_clean
from clusterizer import log

logger = log.get_logger("operations")

log = Logger('baker')


def just_first_doc(docs, docspace, docmap):
    cluster_doc = defaultdict(list)

    clids = [get_cluster_id()]
    obj_id = docs.pop(0)
    cluster_doc[clids[0]] = [obj_id]
    first_doc_ndxes = [docmap[obj_id]]

    for doc in docs:
        doc_ndx = docmap[doc]
        proximities = (docspace[doc_ndx, :] * docspace[first_doc_ndxes, :].transpose()).todense()
        cluster_ndx = argmax(proximities)
        clid = clids[cluster_ndx]
        proximity = proximities[0, cluster_ndx]
        logger.debug('[just_first_doc] Doc %s has proximity %s for cluster %s', doc, proximity, clid)
        if proximity < config.MACRO_CLUSTER_THRESHOLD:
            clid = get_cluster_id()
            clids.append(clid)
            first_doc_ndxes.append(doc_ndx)
        cluster_doc[clid].append(doc)

    return cluster_doc


def build_macro_clusters(newdocs, clusters):
    cluster_doc = defaultdict(list)

    if newdocs:
        newdocs_matrix = append_rows(None, [doc.vector_r1 for doc in newdocs])
        newdocs_matrix = set_columns_number(newdocs_matrix, clusters.dictionary_r1.max)

    docmap = Docmap()
    for doc in newdocs:
        docmap.add(doc.id)

    if clusters:
        clusters.centroids_r1 = set_columns_number(clusters.centroids_r1, clusters.dictionary_r1.max)
        proximities = (newdocs_matrix * clusters.centroids_r1.transpose()).todense()

        glue = argmax(proximities, axis=1)
        glued_docs = set()
        for doc_ndx, cl_ndx in enumerate(glue):
            proximity = proximities[doc_ndx, cl_ndx]
            clid = clusters.clmap_r1.by_index(cl_ndx)
            doc = newdocs[doc_ndx]
            logger.debug('[build_macro_clusters] Doc %s has proximity %s for cluster %s', doc, proximity, clid)
            if proximity < config.MACRO_CLUSTER_THRESHOLD:
                continue

            cluster_doc[clid].append(doc)
            glued_docs.add(doc.id)
        rest = [d for d in newdocs if d.id not in glued_docs]
    else:
        rest = newdocs

    if rest:
        first_doc = just_first_doc(rest, newdocs_matrix, docmap)
        cluster_doc.update(first_doc)
        new_clids = set(first_doc)
    else:
        new_clids = set()

    return cluster_doc, new_clids


def join_clusters(clusters, cluster1, cluster2):
    # join R2 matrices
    # ignore dictionary
    r1_features_num = clusters.dictionary_r1.max
    r2_features_num = clusters.dictionary_r2.max
    cluster1.centroids = append_rows(
                            set_columns_number(cluster1.centroids, r2_features_num),
                            cluster2.centroids)
    cluster1.clmap.append(cluster2.clmap)

    cluster1.docspace = append_rows(
                            set_columns_number(cluster1.docspace, r2_features_num),
                            cluster2.docspace)
    cluster1.docmap.append(cluster2.docmap)

    # join r1 matrices
    cluster1.docspace_r1 = append_rows(
                            set_columns_number(cluster1.docspace_r1, r1_features_num),
                            cluster2.docspace_r1)
    cluster1.docmap_r1.append(cluster2.docmap_r1)

    logger.debug('[join_clusters] %s and %s (%s and %s)', cluster1, cluster2, cluster1.id, cluster2.id)
    cluster1.update(cluster2)


def glue_macro(clusters, active_clids):
    updated_clids, removed_clids = set(), set()
    if not clusters or not active_clids:
        return updated_clids, removed_clids

    active_clids = list(active_clids)
    active_ndxes = [clusters.clmap_r1[clid] for clid in active_clids]
    proximities = clusters.centroids_r1[active_ndxes] * clusters.centroids_r1.transpose()

    for ndx1, ndx2 in enumerate(active_ndxes):
        proximities[ndx1, ndx2] = .0

    glue = [tuple(val[0]) for val in argwhere(proximities.todense() >= config.GLUE_THRESHOLD_MACRO)]
    glue = sorted(glue, key=lambda val: proximities[val[0], val[1]], reverse=True)

    for i, j in glue:
        clid1 = active_clids[i]
        clid2 = clusters.clmap_r1.by_index(j)
        assert clid1 != clid2

        if clid1 not in clusters or clid2 not in clusters:
            continue

        cluster1 = clusters[clid1]
        cluster2 = clusters[clid2]
        if cluster1 and cluster2:

            if getattr(cluster1, 'fixed_centroid', False) or getattr(cluster2, 'fixed_centroid', False):
                continue

            if len(cluster1) < len(cluster2):
                cluster1, cluster2 = cluster2, cluster1

            join_clusters(clusters, cluster1, cluster2)

            cluster1.status = 'updated'
            logger.debug('[glue_macro] Cluster %s delete from storage', cluster2.id)
            del clusters[cluster2.id]
            updated_clids.add(cluster1.id)
            removed_clids.add(cluster2.id)

    updated_clids = set(clid for clid in updated_clids if clid in clusters)

    if updated_clids or removed_clids:
        common.update_centroids_r1(clusters, updated_clids, removed_clids)

    return updated_clids, removed_clids


def just_first(obj_ids, docspace, docmap):
    cluster_doc = defaultdict(list)

    clids = [get_cluster_id()]
    obj_id = obj_ids.pop(0)
    cluster_doc[clids[0]] = [obj_id]
    first_doc_ndxes = [docmap[obj_id]]

    for oid in obj_ids:
        doc_ndx = docmap[oid]
        proximities = (docspace[doc_ndx, :] * docspace[first_doc_ndxes, :].transpose()).todense()
        cluster_ndx = argmax(proximities)
        proximity = proximities[0, cluster_ndx]

        if proximity >= config.MACRO_CLUSTER_THRESHOLD:
            clid = clids[cluster_ndx]
        else:
            clid = get_cluster_id()
            clids.append(clid)
            first_doc_ndxes.append(doc_ndx)
        cluster_doc[clid].append(oid)

    return cluster_doc


def move_micro(clusters, macro1, macro2, micro_clids):
    """
    Move micro clids between macro clusters.
    Actually - copy docspace, but move micro clusters from one to other dict.

        macro1   - Src macro
        macro2   - Dest macro
        micro_clids - Micro clusters to move
    """

    r1_features_num = clusters.dictionary_r1.max
    r2_features_num = clusters.dictionary_r2.max

    docids = []
    for clid in micro_clids:
        docids.extend(map(attrgetter('id'), macro1[clid]))

    # copy R1
    docndxes = map(macro1.docmap_r1.__getitem__, docids)
    macro2.docspace_r1 = append_rows(
                            set_columns_number(macro2.docspace_r1, r1_features_num),
                            macro1.docspace_r1[docndxes, :])
    for docid in docids:
        macro2.docmap_r1.add(docid)

    # copy R2
    docndxes = map(macro1.docmap.__getitem__, docids)
    macro2.docspace = append_rows(
                            set_columns_number(macro2.docspace, r2_features_num),
                            macro1.docspace[docndxes, :])
    for docid in docids:
        macro2.docmap.add(docid)

    # copy R2 centroids
    clndxes = map(macro1.clmap.__getitem__, micro_clids)
    macro2.centroids = append_rows(
                            set_columns_number(macro2.centroids, r2_features_num),
                            macro1.centroids[clndxes, :])
    # adds ids to macro2.clmap, same order as macro2.centroids
    for clid in micro_clids:
        macro2.clmap.add(clid)

    # finally, move clusters
    macro2.update((clid, macro1[clid]) for clid in micro_clids)

    # update doc_cluster
    for clid in micro_clids:
        clusters.doc_cluster.update((docid, (macro2.id, doc.id)) for doc in macro2[clid])
    common.update_centroids_r1(clusters, [macro2.id], [])


def remove_macro(clusters, macro_id):
    common.update_centroids_r1(clusters, [], [macro_id])
    free_micro = clusters[macro_id]
    result = defaultdict(list)

    # compute R1 micro centroids
    clids, micro_centroids = common.build_centroids(free_micro, free_micro.docspace_r1, free_micro.docmap_r1)
    micro_centroids = set_columns_number(micro_centroids, clusters.dictionary_r1.max)
    clmap = Docmap()
    for clid in clids:
        clmap.add(clid)

    if clusters:
        clusters.centroids_r1 = set_columns_number(clusters.centroids_r1, clusters.dictionary_r1.max)
        proximities = (micro_centroids * clusters.centroids_r1.transpose()).todense()

        glue = argmax(proximities, axis=1)
        glued_micro = set()
        for micro_ndx, cl_ndx in enumerate(glue):
            proximity = proximities[micro_ndx, cl_ndx]
            if proximity < config.MACRO_CLUSTER_THRESHOLD:
                continue

            micro_clid = clids[micro_ndx]
            clid = clusters.clmap_r1.by_index(cl_ndx)
            glued_micro.add(clid)
            result[clid].append(micro_clid)

        rest = [clid for clid in free_micro.keys() if clid not in glued_micro]
    else:
        rest = free_micro.keys()

    if rest:
        first_doc = just_first(rest, micro_centroids, clmap)
        for clid in first_doc.keys():
            clusters[clid] = MacroCluster(clid)

        result.update(first_doc)

    for macro_id2, micro_clids in result.items():
        move_micro(clusters, free_micro, clusters[macro_id2], micro_clids)

    del clusters[macro_id]


def update_manual_mapping(clusters, baker_data):
    "Save such mapping: {macro_id: [micro_id1, micro_id1, micro_id1, ...]}"
    mapping = defaultdict(list)
    for clid1, clid2 in clusters.doc_cluster.itervalues():
        mapping[clid1].append(clid2)

    manual_clids = baker_data['manual_clusters'].keys()
    baker_data['manual_mapping'] = dict((clid, mapping[clid]) for clid in manual_clids)


def load_manual_updates(clusters, upd_clids, baker_data):
    # FIXME: words should be lemmatized before
    clusters.preload(upd_clids)
    updated_centroids = []

    manual_clusters = baker_data['manual_clusters']

    for clid in upd_clids:
        if clid not in manual_clusters:
            if clid in clusters and getattr(clusters[clid], 'fixed_centroid', False):
                remove_macro(clusters, clid)
                if clid in baker_data['manual_r1_terms']:
                    del baker_data['manual_r1_terms'][clid]
            continue

        try:
            terms = list(lemmatize_clean(manual_clusters[clid]['words'].decode('utf-8')))
        except UnicodeEncodeError as e:
            continue

        if clid not in clusters:
            cluster = MacroCluster(clid)
            cluster.fixed_centroid = True
            clusters[clid] = cluster

        baker_data['manual_r1_terms'][clid] = terms

        data = [1.0] * len(terms)
        row_coords = [0] * len(terms)
        col_coords = map(clusters.dictionary_r1.add, terms)

        centroid = coo_matrix((data, (row_coords, col_coords)),
                      shape=(1, clusters.dictionary_r1.max))
        updated_centroids.append((clid, centroid))

    if updated_centroids:
        clusters.centroids_r1 = update_docspace_helper(clusters.clmap_r1,
            clusters.centroids_r1, updated_centroids, [], clusters.dictionary_r1.max)


def clean(clusters, updated_clids, bd):
    if updated_clids:
        micro_mtime = bd['micro_mtime']
        curr = int(time.time())
        micro_mtime.update((micro_id, curr) for macro_id, micro_id in updated_clids)

    cut_date = datetime.now() - config.CLUSTER_ACTIVE_TIMEDELTA
    cut_date = time.mktime(cut_date.timetuple())
    micro_mtime = bd['micro_mtime']
    outdated = [clid for clid, mtime in micro_mtime.iteritems() if mtime < cut_date]
    for clid in outdated:
        del micro_mtime[clid]

    mapping = {}
    for (macro_id, micro_id) in clusters.doc_cluster.values():
        mapping[micro_id] = macro_id

    outdated = [clid for clid in outdated if clid in mapping]
    macro_ids = map(mapping.__getitem__, outdated)
    clusters.preload(macro_ids)

    removed_docs = []
    for clid1, clid2 in zip(macro_ids, outdated):
        removed_docs.extend(map(attrgetter('id'), clusters[clid1][clid2]))

    macro_updated = remove_docs_twolevel(clusters, removed_docs)[0]
    logger.debug('[clean] removed docs: %s', removed_docs)
    # FIXME:
    update_centroids_r1(clusters, macro_updated, [])
    return removed_docs

