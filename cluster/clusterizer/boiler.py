import copy

from clusterizer.cluster import MacroCluster
from clusterizer import algorithms
from clusterizer import common
from clusterizer import config
from clusterizer import log
from clusterizer import macro_operations
from clusterizer import micro_operations

logger = log.get_logger("boiler")


def boil_hier(clusters, new_docs):
    all_docs = []
    for cl in clusters.values():
        all_docs.extend(cl)

    all_docs.extend(new_docs)
    pseudo_clusters = algorithms.hierarchical(clusters.docspace, clusters.docmap, all_docs)
    pseudo_clusters = dict(enumerate(pseudo_clusters))
    updated_clids, removed_clids = micro_operations.reassign_helper(clusters, clusters.keys(),  pseudo_clusters)

    if updated_clids or removed_clids:
        common.update_centroids(clusters, updated_clids, removed_clids)
    if updated_clids:
        common.update_main_docs(clusters, updated_clids)

    return clusters, updated_clids, removed_clids, {}


def boil_kmeans(clusters, new_docs, active_clids):
    removed_clids = set()
    updated_clids = set(active_clids)

    # recompute clusters with removed items
    if updated_clids:
        common.update_centroids(clusters, updated_clids, removed_clids)
        common.update_main_docs(clusters, updated_clids)

    updated_clids0, active_thres_clids = micro_operations.build_clusters(new_docs, clusters)
    updated_clids |= updated_clids0

    updated_clids1, removed_clids1 = algorithms.kmeans(clusters, (updated_clids | active_thres_clids))

    updated_clids = (updated_clids | updated_clids1) - removed_clids1
    removed_clids |= set(removed_clids1)

    merged_clids = dict()
    for i in range(config.GLUE_ITERATIONS):
        updated_clids2, removed_clids2, merged_clids2 = micro_operations.glue_clusters(
            clusters=clusters,
            active_clids=updated_clids,
            method=config.GLUE_METHOD
        )
        if not updated_clids2:
            break
        updated_clids = (updated_clids | updated_clids2) - removed_clids2
        removed_clids |= set(removed_clids2)
        merged_clids.update(merged_clids2)

    logger.debug('Boil kmeans %s', clusters)
    logger.debug('Removed clids %s', removed_clids)
    logger.debug('Merged clids %s', merged_clids)
    return clusters, updated_clids, removed_clids, merged_clids


def boil_micro(micro_clusters, new_docs, active_clids):
    for cluster in micro_clusters.itervalues():
        cluster.status = ''

    if config.CLUSTER_METHOD == 'hier':
        clusters, updated_clids, removed_clids, merged_clids = boil_hier(micro_clusters, new_docs, active_clids)
    elif config.CLUSTER_METHOD == 'kmeans':
        clusters, updated_clids, removed_clids, merged_clids = boil_kmeans(micro_clusters, new_docs, active_clids)
    else:
        raise ValueError('CLUSTER_METHOD is defined incorrectly')

    if updated_clids:
        for clid in updated_clids:
            clusters[clid].status = 'updated'

        micro_operations.process_clusters(clusters, updated_clids)

    return clusters, updated_clids, removed_clids, merged_clids


def boil_macro(clusters, prepared_docs, macro_updated):
    cluster_doc, new_clids = macro_operations.build_macro_clusters(prepared_docs, clusters)
    updated_macro_clids = set(macro_updated.keys() + cluster_doc.keys())

    for clid in updated_macro_clids:
        docs = copy.deepcopy(cluster_doc.get(clid, []))
        micro_clids = macro_updated.get(clid, [])
        if clid in new_clids:
            macro_cluster = MacroCluster(clid)
            clusters[clid] = macro_cluster
        else:
            macro_cluster = clusters[clid]

        updated_rows_r1, updated_rows_r2 = [], []
        for doc in docs:
            updated_rows_r1.append((doc.id, doc.vector_r1))
            updated_rows_r2.append((doc.id, doc.vector_r2))
            del doc.vector_r1
            del doc.vector_r2

        macro_cluster.docspace_r1 = common.update_docspace_helper(
            docmap=macro_cluster.docmap_r1,
            matrix=macro_cluster.docspace_r1,
            updated_rows=updated_rows_r1,
            removed_ids=[],
            features_num=clusters.dictionary_r1.max
        )
        macro_cluster.docspace = common.update_docspace_helper(
            docmap=macro_cluster.docmap,
            matrix=macro_cluster.docspace,
            updated_rows=updated_rows_r2,
            removed_ids=[],
            features_num=clusters.dictionary_r2.max
        )

        macro_cluster.features_num = clusters.dictionary_r2.max
        merged_clids = boil_micro(macro_cluster, docs, micro_clids)[3]
        clusters.merged_clids.update(merged_clids)

    common.update_centroids_r1(clusters, updated_macro_clids, [])

    updated_macro_clids2, removed_macro_clids = macro_operations.glue_macro(clusters, updated_macro_clids)
    updated_macro_clids |= updated_macro_clids2
    updated_macro_clids -= removed_macro_clids

    for macro_id in updated_macro_clids:
        macro = clusters[macro_id]
        for micro in macro.itervalues():
            clusters.doc_cluster.update((d.id, (macro.id, micro.id)) for d in micro)

    return updated_macro_clids, new_clids, removed_macro_clids
