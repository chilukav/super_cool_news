# -*- coding: utf-8 -*-
import collections

from numpy import argmax
from numpy import clip, unravel_index
from scipy.cluster.hierarchy import fcluster, inconsistent, linkage
from scipy.spatial import distance

from cluster.clusterizer import common
from cluster.clusterizer import config
from cluster.clusterizer import projects
from cluster.clusterizer.cluster import Cluster, get_cluster_id
from cluster.clusterizer import log

logger = log.get_logger("algorithms")


def single_doc_mapping(clusters):
    return dict((cl[0].id, clid) for clid, cl in clusters.iteritems() if len(cl) == 1)


def _kmeans(clusters, docspace, docmap, iterations, clid_creation):

    doc_cluster = []
    for clid, cldocs in clusters.iteritems():
        doc_cluster.extend((d.id, clid) for d in cldocs)
    doc_cluster = dict(doc_cluster)

    docs = []
    for c in clusters.itervalues():
        docs.extend([(docmap[d.id], d) for d in c])

    updated_clids = []

    for i in range(iterations):
        if i > 0 and not updated_clids:
            break

        clids, centroids = common.build_centroids(clusters, docspace, docmap)

        proximities = common.proximity_matrix(centroids, docspace).todense()
        proximities = projects.patch_proximities(docs, clids, clusters, proximities, new_docs=False)

        single_doc = single_doc_mapping(clusters)
        new_clusters = collections.defaultdict(list)
        new_doc_cluster = []
        for doc_ndx, doc in docs:
            centroid_ndx = argmax(proximities[:, doc_ndx])
            cluster_id = clids[centroid_ndx]
            max_proximity = proximities[centroid_ndx, doc_ndx]

            if not clid_creation or max_proximity >= config.CLUSTER_THRESHOLD:
                doc.proximity = max_proximity

            else:
                doc.proximity = 1.0
                cluster_id = single_doc[doc.id] if doc.id in single_doc else get_cluster_id()

            new_clusters[cluster_id].append(doc)
            new_doc_cluster.append((doc.id, cluster_id))
            logger.debug('[_kmeans] Add doc %s to cluster %s proximity=%s', doc, cluster_id, doc.proximity)

        updated_clids = [clid for docid, clid in new_doc_cluster if clid != doc_cluster[docid]]
        clusters = new_clusters
        doc_cluster = dict(new_doc_cluster)

    return clusters


def kmeans(clusters, clids):
    clids = set(clids)

    if len(clids) <= 1 and not config.KMEANS_CREATES_CLUSTER:
        return set(), set()

    doc_mapping = dict()
    for clid in clids:
        doc_mapping[clid] = clusters[clid][:]

    doc_mapping = _kmeans(
        clusters=doc_mapping,
        docspace=clusters.docspace,
        docmap=clusters.docmap,
        iterations=config.KMEANS_ITERATIONS,
        clid_creation=config.KMEANS_CREATES_CLUSTER
    )

    removed_clids = clids - set(doc_mapping.keys())

    if removed_clids:
        logger.debug("[kmeans] %s clusters glued.", len(removed_clids))

    updated_clids = []
    for clid, docs in doc_mapping.iteritems():
        if clid not in clusters:
            clusters[clid] = Cluster(clid)
        else:
            cluster = clusters[clid]
            if sorted(cluster[:]) == sorted(docs):
                continue
        clusters[clid][:] = docs
        updated_clids.append(clid)

    for clid in removed_clids:
        try:
            del clusters[clid]
        except KeyError:
            pass

    if updated_clids:
        common.update_centroids(clusters, updated_clids, removed_clids)
        common.update_main_docs(clusters, updated_clids)

    return set(updated_clids), removed_clids


def hierarchical(docspace, docmap, newdocs, metric='cosine',
               link_method='complete', threshold=config.HIERARCHICAL_THRESHOLD):
    if len(newdocs) == 1:
        return [newdocs]
    doc_ndxes = [docmap[doc.id] for doc in newdocs]
    Y = distance.pdist(docspace[doc_ndxes, :].todense(), metric=metric)
    # Due to machine precision some zero values occur negative, fix this
    Y = clip(Y, 0, 1)
    # Merge similar clusters with the given linkage function
    Z = linkage(Y, method=link_method, metric=metric)
    assignment = fcluster(Z, criterion='distance', t=threshold)
    pseudo_clusters = [[] for _ in range(max(assignment))]
    for i, j in enumerate(assignment):
        pseudo_clusters[j - 1].append(newdocs[i])
    return pseudo_clusters


def min_distance(distances, real_clusters, pseudo_clusters, method='argmin'):
    distance_dict = {}
    for pseudo_ndx, pseudo_cluster in enumerate(pseudo_clusters):
        for cluster_ndx, cluster in enumerate(real_clusters):
            rows = [[e] for e in cluster]
            cols = pseudo_cluster
            matrix = distances[rows, cols]
            i, j = unravel_index(getattr(matrix, method)(), matrix.shape)
            distance_dict[matrix[i, j]] = [cluster_ndx, pseudo_ndx]

    if not distance_dict:
        return None, [None, None]
    min_distance = min(distance_dict)
    return min_distance, distance_dict[min_distance]


def join(real_clusters, pseudo_clusters, docspace, docmap):
    added_updated_clids = set()
    pseudo_clusters_ndxes = []
    for pseudo_cluster in pseudo_clusters:
        pseudo_clusters_ndxes.append([])
        for doc in pseudo_cluster:
            pseudo_clusters_ndxes[-1].append(docmap[doc.id])

    clusters_mapping = []
    clusters = []
    for clid, cluster in real_clusters.iteritems():
        clusters_mapping.append(clid)
        clusters.append([])
        for doc in cluster:
            clusters[-1].append(docmap[doc.id])

    distances = 1 - (docspace * docspace.T).todense()
    pseudo_clusters_to_create = []
    if not clusters:
        pseudo_clusters_to_create = pseudo_clusters
        pseudo_clusters = []
    while pseudo_clusters:
        min_dist, [cluster_ndx, pseudo_cluster_ndx] =\
            min_distance(distances, clusters, pseudo_clusters_ndxes)
        if min_dist <= config.HIERARCHICAL_THRESHOLD:
            clid = clusters_mapping[cluster_ndx]
            real_clusters[clid].extend(pseudo_clusters[pseudo_cluster_ndx])
            clusters[cluster_ndx].extend(pseudo_clusters_ndxes[pseudo_cluster_ndx])
            #del clusters_mapping[cluster_ndx]
            #del clusters[cluster_ndx]
            added_updated_clids.add(clid)
        else:
            pseudo_cluster = pseudo_clusters[pseudo_cluster_ndx]
            pseudo_clusters_to_create.append(pseudo_cluster)

        del pseudo_clusters[pseudo_cluster_ndx]
        del pseudo_clusters_ndxes[pseudo_cluster_ndx]

    for pseudo_cluster in pseudo_clusters_to_create:
        cluster = Cluster(get_cluster_id())
        cluster.extend(pseudo_cluster)
        real_clusters[cluster.id] = cluster
        added_updated_clids.add(cluster.id)

    return added_updated_clids

