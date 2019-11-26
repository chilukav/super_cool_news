from munkres import Munkres
from numpy import argwhere
from numpy import argmax
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

from cluster.clusterizer import common
from cluster.clusterizer import config
from cluster.clusterizer import projects
from cluster.clusterizer.cluster import Cluster, get_cluster_id
from cluster.clusterizer.metrics import fm
from cluster.clusterizer.utils.matrix import set_columns_number
from cluster.clusterizer import log

logger = log.get_logger("micro_operations")


def build_clusters(newdocs, clusters):
    clids = []
    first_doc_ndxes = []
    new_ndx = {}
    active_clids = set()
    updated_clids = set()
    docspace, docmap = clusters.docspace, clusters.docmap
    centroids = set_columns_number(clusters.centroids, clusters.features_num)

    for _, clid in clusters.clmap:
        cluster = clusters[clid]
        clids.append(cluster.id)
        ndx = docmap[cluster[0].id]
        new_ndx[ndx] = len(first_doc_ndxes)
        first_doc_ndxes.append(ndx)

    for doc in newdocs:
        if doc.id not in docmap: continue
        doc_ndx = docmap[doc.id]
        proximity = 0
        if clids:
            if centroids is not None:
                proximities = (docspace[doc_ndx, :] * centroids.T).todense()
            else:
                proximities = (docspace[doc_ndx, :] * docspace[first_doc_ndxes, :].transpose()).todense()

            proximities = projects.patch_proximities([(0, doc)], clids, clusters, proximities.transpose(), new_docs=True)
            proximities = proximities.transpose()

            active_thres = argwhere(proximities >= config.ACTIVE_THRESHOLD)
            active_thres = set(clids[m[0, 1]] for m in active_thres)
            active_clids |= active_thres

            cluster_ndx = argmax(proximities)
            proximity = proximities[0, cluster_ndx]

        if proximity >= config.FIRST_DOC_THRESHOLD:
            doc.proximity = proximity
            cluster = clusters[clids[cluster_ndx]]
        else:
            proximity = -1
            cluster = Cluster(get_cluster_id())
            clusters[cluster.id] = cluster
            clids.append(cluster.id)
            first_doc_ndxes.append(doc_ndx)
            if centroids is not None:
                centroids = vstack([centroids, docspace[doc_ndx, :]], format='csr')

        cluster.append(doc)
        updated_clids.add(cluster.id)
        logger.debug('[build_clusters] Add doc %s to micro %s proximity=%s', doc, cluster, proximity)

        if doc.is_main_resource:
            cluster.manual_pitem = doc.id

    # workaround - to add new centroids to clmap and centroids matrix
    # actually - there is no need in this, cause we will recompute centroids in kmeans
    if updated_clids:
        common.update_centroids(clusters, updated_clids, [])
        common.update_main_docs(clusters, updated_clids)

    return updated_clids, active_clids


def prepare(clusters, clids=None):
    res = []
    idx_mapping = {}

    if clids:
        clusters_subset = [(clid, clusters[clid]) for clid in clids]
    else:
        clusters_subset = clusters.items()

    for clid, cl in clusters_subset:
        res.append(set(doc.id for doc in cl))
        idx_mapping[len(res) - 1] = clid
    return idx_mapping, res


def reassign(original_clusters, active_orig_clids, new_clusters):
    orig_idx, original_clusters = prepare(original_clusters, active_orig_clids)
    new_idx, new_clusters = prepare(new_clusters)
    matrix = []
    for ncl in new_clusters:
        matrix.append([1.0 - fm(0.5, ocl, ncl) for ocl in original_clusters])

    indexes = Munkres().compute(matrix)

    notchanged = {}
    mapping = {}
    for row, column in indexes:
        if new_clusters[row] == original_clusters[column]:
            notchanged[new_idx[row]] = orig_idx[column]
        else:
            mapping[new_idx[row]] = orig_idx[column]
    return notchanged, mapping


def reassign_helper(old_clusters, old_active_clids, new_clusters):
    """
    Match new clusters to old one,
    create new clids, if necessary,
    and remove not matched old clusters
    """

    notchanged, mapping = reassign(old_clusters, old_active_clids, new_clusters)
    matched_old_clids = set(mapping.values() + notchanged.values())

    removed_clids = set()
    for clid in old_active_clids:
        if clid not in matched_old_clids:
            del old_clusters[clid]
            removed_clids.add(clid)

    updated_clids = set()
    for idx, new_cluster in new_clusters.items():
        if idx in notchanged:
            continue
        if idx in mapping:
            cluster = old_clusters[mapping[idx]]
        else:
            clid = get_cluster_id()
            old_clusters[clid] = cluster = Cluster(clid)

        cluster[:] = new_cluster[:]
        updated_clids.add(cluster.id)
    return updated_clids, removed_clids


def glue_allowed(cluster1, cluster2, docspace, docmap):
    if getattr(cluster1, 'has_meta', None) and getattr(cluster2, 'has_meta', None):
        logger.debug('[glue_disallowed] %s and %s have meta', cluster1, cluster2)
        return False

    if not projects.projects_check(cluster1, cluster2) or not projects.main_check(cluster1, cluster2):
        logger.debug(
            '[glue_disallowed] %s and %s have different projects or both with main resources',
            cluster1,
            cluster2
        )
        return False

    return try_glue(cluster1, cluster2, docspace, docmap)


def try_glue(cluster1, cluster2, docspace, docmap):
    # this simply doesn't work! - cluster may glue into 2 or more other clusters
    # should be tring a general kmeans with less centroids
    docs = cluster1 + cluster2
    idxes = [docmap[doc.id] for doc in docs]
    centroid = csr_matrix(common.build_centroid(idxes, docspace))
    prox = docspace[idxes, :] * centroid.T
    r = argwhere(prox.todense() < config.CLUSTER_THRESHOLD)
    logger.debug('[try_glue] cluster1 %s and cluster2 %s: %s', cluster1.id, cluster2.id, r)
    return len(r) == 0


def reverse_glue_order(cluster1, cluster2):
    """
        Returns True if the glue order should be reversed,
        i.e. cluster1 should be included in cluster2.
    """
    c1_has_meta = getattr(cluster1, 'has_meta', None)
    c2_has_meta = getattr(cluster2, 'has_meta', None)
    if c1_has_meta or c2_has_meta:
        # reverse, if cluster2 has meta, and cluster1 don't.
        if c2_has_meta and not c1_has_meta:
            return True

        if c1_has_meta and not c2_has_meta:
            return False
        # if they both has meta - then such glue should be blocked


    c1_manual_pos = getattr(cluster1, 'manual_pos', None)
    c2_manual_pos = getattr(cluster2, 'manual_pos', None)
    if c1_manual_pos or c2_manual_pos:
        # reverse, if cluster2 has manual pos, and cluster1 don't.
        if c2_manual_pos and not c1_manual_pos:
            return True

        if c1_manual_pos and not c2_manual_pos:
            return False
        # if they both has manual pos - follow standard logic


    if hasattr(cluster1, 'ctime') and hasattr(cluster2, 'ctime'):
        # move items from newer to older cluster
        return cluster1.ctime > cluster2.ctime
    else:
        return len(cluster1) < len(cluster2)

    return False


def glue_clusters(clusters, active_clids, method):
    docspace, docmap = clusters.docspace, clusters.docmap

    if method == 'CENTROID':
        clids = clusters.keys()
        matrix = common.clusters_centroids(clusters, clids)
    elif method == 'MAIN':
        clids = []
        main_ndxes = []
        for cluster in clusters.itervalues():
            clids.append(cluster.id)
            main_ndxes.append(docmap[cluster[0].id])
        matrix = docspace[main_ndxes, :]
    else:
        raise Exception('Unknown glue method')

    active_clids = list(active_clids)
    active_ndxes = [clids.index(clid) for clid in active_clids]
    proximities = matrix[active_ndxes] * matrix.transpose()

    for ndx, clid in enumerate(active_clids):
        ndx2 = clids.index(clid)
        proximities[ndx, ndx2] = .0

    glue = argwhere(proximities.todense() >= config.GLUE_THRESHOLD)
    updated_clids, removed_clids, merged_clids = set(), set(), dict()
    for r in glue:
        clid1 = active_clids[r[0, 0]]
        clid2 = clids[r[0, 1]]

        if clid1 not in clusters or clid2 not in clusters:
            continue

        cluster1 = clusters[clid1]
        cluster2 = clusters[clid2]
        if cluster1 and cluster2:

            if reverse_glue_order(cluster1, cluster2):
                cluster1, cluster2 = cluster2, cluster1

            if not glue_allowed(cluster1, cluster2, docspace, docmap):
                continue

            for doc in cluster2:
                cluster1.append(doc)

            updated_clids.add(cluster1.id)
            removed_clids.add(cluster2.id)
            del clusters[cluster2.id]
            merged_clids[cluster2.id] = cluster1.id
            logger.debug('[glue_clusters] Merge clusters %s and %s', cluster1.id, cluster2.id)

    updated_clids -= removed_clids

    if updated_clids or removed_clids:
        common.update_centroids(clusters, updated_clids, removed_clids)

    if updated_clids:
        common.update_main_docs(clusters, updated_clids)

    return updated_clids, removed_clids, merged_clids


def process_clusters(clusters, clids):
    new_idxes = {}
    docs_ndxes = []
    docmap, docspace = clusters.docmap, clusters.docspace

    for clid in clids:
        for doc in clusters[clid]:
            new_idxes[doc.id] = len(docs_ndxes)
            docs_ndxes.append(docmap[doc.id])

    # Get proximity matrix and set final proximities
    ds_part = docspace[docs_ndxes]
    proximities = common.proximity_matrix(ds_part, ds_part)

    for clid in clids:
        cluster = clusters[clid]

        if len(cluster) <= 1:
            continue
        docs = cluster[1:]  # don't touch main_doc
        docs_proximities = [proximities[new_idxes[cluster[0].id], new_idxes[doc.id]] for doc in docs]
        docs_proximities = common.complex_proximities(docs, docs_proximities)

        for doc, prox in zip(docs, docs_proximities):
            doc.proximity = prox
            if doc.proximity >= config.KERNEL_THRESHOLD:
                doc.status = 'kernel'
            else:
                doc.status = 'work'

        cluster.sort()
