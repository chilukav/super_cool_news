# TODO remove me



import datetime as dt
import cPickle
import os
import re
from glob import glob

from django.conf import settings

from scraper.redis import RedisHash
from scraper.clusterizer.cluster import MacroClusters
from scraper.utils import Logger


ALL_CLUSTERS_HASH = 'very_very_super_clusters_storage'

log = Logger('baker_storage')


class AllClustersStorage(MacroClusters):
    SELF_STORAGE_KEY = 'all_macro_clusters'

    def __init__(self):
        self.removed_clids = set()
        self.merged_clids = dict()
        r = RedisHash(ALL_CLUSTERS_HASH)
        stored_attributes = r.get(self.SELF_STORAGE_KEY)
        macro_clids = [int(clid) for clid in r.get_keys() if clid != self.SELF_STORAGE_KEY]

        if stored_attributes is not None and macro_clids is not None:
            self.__dict__.update(stored_attributes)
            for k in macro_clids:
                MacroClusters.__setitem__(self, k, None)
        else:
            MacroClusters.__init__(self)

    def clean_storage(self):
        RedisHash(ALL_CLUSTERS_HASH).clean_all()

    def preload(self, clids):
        clids_to_load = [clid for clid in clids if clid in self and not self.loaded(clid)]
        if clids_to_load:
            clids_to_load = list(set(clids_to_load))
            values = RedisHash(ALL_CLUSTERS_HASH).mget(clids_to_load)
            self.update(zip(clids_to_load, values))

    def loaded(self, clid):
        value = MacroClusters.__getitem__(self, clid)
        return (value is not None)

    def __getitem__(self, clid):
        try:
            value = MacroClusters.__getitem__(self, clid)
            if value is not None:
                return value
            else:
                raise KeyError((clid,))
        except KeyError:
            if clid in self.removed_clids:
                raise
            value = self.load_macro_cluster(clid)
            if value is not None:
                self[clid] = value
                return value
            else:
                raise

    def __setitem__(self, clid, macro):
        self.removed_clids.discard(clid)
        MacroClusters.__setitem__(self, clid, macro)

    def __delitem__(self, clid):
        # TODO: update doc_cluster
        self.removed_clids.add(clid)
        MacroClusters.__delitem__(self, clid)

    def sync(self):
        # separate clusters into macro_cluster items and
        #  macro_clusters structure without any items

        rhash = RedisHash(ALL_CLUSTERS_HASH)

        rhash.client = rhash.client.pipeline(transaction=True)
        updated_clusters = dict((k, v) for k, v in self.macro_clusters.items() if v is not None)
        if updated_clusters:
            for k, v in updated_clusters.iteritems():
                rhash.set(k, v)

        if self.removed_clids:
            rhash.delete_keys(self.removed_clids)

        self.removed_clids = set()

        # workaround - temporary remove all macro clusters, to store only attributes
        back_up = self.macro_clusters
        self.macro_clusters = dict()
        rhash.set(self.SELF_STORAGE_KEY, self.__dict__)

        rhash.client.execute()

        # restore removed - to avoid redundant Redis requests
        self.macro_clusters = back_up

    def save(self, filename):
        # self.preload(self.keys())
        with open(filename, 'wb') as fo:
            cPickle.dump(self, fo, -1)

    @staticmethod
    def load_to_redis(filename):
        with open(filename, 'rb') as fo:
            clusters = cPickle.load(fo)
        RedisHash(ALL_CLUSTERS_HASH).clean_all()
        clusters.sync()
        return clusters

    def load_many_macro_clusters(self, clids):
        return RedisHash(ALL_CLUSTERS_HASH).mget(clids)

    def load_macro_cluster(self, clid):
        return RedisHash(ALL_CLUSTERS_HASH).get(clid)


def check_consistency():
    "Check clusters consistency"
    clusters = AllClustersStorage()
    clusters.preload(clusters.keys())
    err_desc = ""

    elements = []
    for m1 in clusters.values():
        for m2 in m1.values():
            elements.extend(m2)

    if sorted(clusters.doc_cluster.keys()) != sorted([d.id for d in elements]):
        s_dc = set(clusters.doc_cluster.keys())
        s_el = set([d.id for d in elements])
        err_desc += "set(doc_cluster) - set(elements)=" + str(s_dc - s_el) + "\n"
        err_desc += "set(elements) - set(doc_cluster)=" + str(s_el - s_dc) + "\n"

    ne_macroid = set()
    ne_microid = set()
    problem_items = []
    for docid, (clid1, clid2) in clusters.doc_cluster.items():
        try:
            macro = clusters[clid1]
        except KeyError:
            ne_macroid.add(clid1)
            problem_items.append(docid)
            continue

        try:
            micro = macro[clid2]
        except KeyError:
            ne_microid.add(clid2)
            problem_items.append(docid)
            continue

        if not micro.count(docid):
            problem_items.append(docid)

    if problem_items or ne_microid or ne_macroid:
        err_desc += "p_items:{0}, ne_macro:{1}, ne_micro:{2}\n"\
                        .format(problem_items, ne_macroid, ne_microid)

    for m1 in clusters.values():

        docids = []
        for m2 in m1.values():
            for d in m2:
                docids.append(d.id)

        # dm  r2
        if sorted(docids) != sorted(m1.docmap.ndx_doc) \
            or sorted(docids) != sorted(m1.docmap.doc_ndx.keys()):
            err_desc += "macro {0} r2: {1}\n".format(m1.id, set(docids) ^ set(m1.docmap.ndx_doc))

        # dm  r1
        if sorted(docids) != sorted(m1.docmap_r1.ndx_doc) \
            or sorted(docids) != sorted(m1.docmap_r1.doc_ndx.keys()):
            err_desc += "macro {0} r1: {1}\n".format(m1.id, set(docids) ^ set(m1.docmap_r1.ndx_doc))

        # ds
        if m1.docspace is not None and m1.docspace_r1 is not None:
            if len(docids) != m1.docspace.shape[0] \
                or len(docids) != m1.docspace_r1.shape[0]:
                err_desc += "macro {0} shapes: {1} {2} {3}\n" \
                    .format(m1.id, len(docids), m1.docspace.shape[0], m1.docspace_r1.shape[0])
        else:
            if len(docids) > 0:
                err_desc += "macro {0} shapes: {1}, while docspace is None\n" \
                    .format(m1.id, len(docids))

        # clmap
        if sorted(m1.keys()) != sorted(m1.clmap.ndx_doc) \
            or sorted(m1.keys()) != sorted(m1.clmap.doc_ndx.keys()):
            err_desc += "macro {0} clmap: {1}\n" \
                .format(m1.id, set(docids) ^ set(m1.clmap.ndx_doc))

    # clmap
    if sorted(clusters.keys()) != sorted(clusters.clmap_r1.ndx_doc) \
        or sorted(clusters.keys()) != sorted(clusters.clmap_r1.doc_ndx.keys()):
        err_desc += "global clmap: {0}\n" \
                .format(set(clusters.keys()) ^ set(clusters.clmap_r1.ndx_doc))

    log.bark('[check_consistency]: {}'.format(err_desc or 'no errors'))
    return (not err_desc), clusters


def remove_old_backups():
    files = glob(os.path.join(settings.BAKER_BACKUP_DIR, "[0-9]*.pickle"))
    stamps = [re.match("(\d+)\.pickle", os.path.basename(path)).group(1) \
                for path in files]

    if stamps:
        stamps = sorted(map(int, stamps), reverse=True)
        to_remove = stamps[settings.BACKUP_LIMIT:]

        for stamp in to_remove:
            filename = os.path.join(settings.BAKER_BACKUP_DIR, str(stamp) + ".pickle")
            os.remove(filename)


def dump(clusters):
    "Dump clusters structure to file"

    timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = os.path.join(settings.BAKER_BACKUP_DIR, timestamp + ".pickle")

    clusters.save(filename)
    remove_old_backups()


def restore():
    "Restore clusters structure from file to Redis"


    files = glob(os.path.join(settings.BAKER_BACKUP_DIR, "[0-9]*.pickle"))

    stamps = [re.match("(\d+)\.pickle", os.path.basename(path)).group(1) for path in files]

    if not stamps:
        # well, now we can panic
        raise RuntimeError("clusters is not cosistent, and there is no backup")

    last = max(map(int, stamps))
    # load clusters from file
    filename = os.path.join(settings.BAKER_BACKUP_DIR, str(last) + ".pickle")

    AllClustersStorage.load_to_redis(filename)
