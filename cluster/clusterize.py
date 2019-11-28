from clusterizer import boiler
from clusterizer import docspace
from clusterizer import macro_operations
from clusterizer.cluster import MacroClusters
from clusterizer.classify import classify


def clusterize(new_docs):
    clusters = MacroClusters()
    item_ids = [d['id'] for d in new_docs]

    added_docs, updated_docs, removed_docids, macro_updated = docspace.prepare_docs(
        clusters=clusters,
        item_ids=list(set(item_ids)),
        clean=True,
        baker_data=baker_data
    )

    updated_macro_clids, new_macro_clids, removed_macro_clids = boiler.boil_macro(
        clusters=clusters,
        prepared_docs=added_docs + updated_docs,
        macro_updated=macro_updated
    )

    updated_clids = []
    for macro_id in updated_macro_clids:
        macro = clusters[macro_id]
        for micro in macro.itervalues():
            if micro.status != 'updated':
                continue
            updated_clids.append((macro.id, micro.id))

    new_clids = set(new_macro_clids)
    removed_clids = set(removed_macro_clids)
    log.bark('Clusterized.')

    removed_docids += macro_operations.clean(clusters, updated_clids, baker_data)

