# TODO Useless module

from collections import defaultdict

# TODO ItemProperties is missing
from scraper.redis import ItemProperties


def classify(clusters, updated_clids):
    kern_docids = []
    for clid1, clid2 in updated_clids:
        cluster = clusters[clid1][clid2]
        for doc in cluster:
            if (doc.status == 'kernel' or doc.status == 'main'):
                kern_docids.append(doc.id)

    if kern_docids:
        props = ItemProperties().mget(kern_docids)

        props_dict = {}
        for docid, p in zip(kern_docids, props):
            if p:
                props_dict[docid] = p

        for clid1, clid2 in updated_clids:
            cluster = clusters[clid1][clid2]
            cluster.topics = weighted_sum(cluster, props_dict, 'topics')
            cluster.regions = weighted_sum(cluster, props_dict, 'regions')


def weighted_sum(cluster, props_dict, attr):
    result = defaultdict(float)

    norm = 0.0

    for doc in cluster:
        if (doc.status == 'kernel' or doc.status == 'main') and doc.id in props_dict:
            norm += doc.proximity
            attrvalue = props_dict[doc.id][attr]
            for value, weight in attrvalue:
                result[value] += weight * doc.proximity

    if norm > 1e-10:
        for value in result:
            result[value] /= norm
        return dict(result)
    else:
        return dict()
