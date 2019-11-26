import collections
import datetime as dt
import heapq
import marshal
import pickle

# TODO a log of work here.
# TODO What is main purpose of imports? Do we need them?

from scraper_common.db import lazyload
from scraper.clusterizer import projects
from scraper.redis import ClusterIdSeq


def get_cluster_id():
    return ClusterIdSeq().next_id()


class Dictionary(object):
    def __init__(self):
        self.q = []
        self.max = 0
        self.terms = {}
        self.idx_count = {}
        self.indexes = {}

    def _free_index(self):
        try:
            return heapq.heappop(self.q)
        except IndexError:
            m = self.max
            self.max += 1
            return m

    def __getitem__(self, term):
        return self.terms[term]

    def __contains__(self, term):
        return term in self.terms

    def __repr__(self):
        return repr(self.terms)

    def add(self, term):
        if term in self.terms:
            ndx = self.terms[term]
            self.idx_count[ndx] += 1
            return self.terms[term]

        ndx = self._free_index()
        self.terms[term] = ndx
        self.indexes[ndx] = term
        self.idx_count[ndx] = 1
        return ndx

    def remove(self, term, count=1):
        if term not in self.terms:
            return

        ndx = self.terms[term]
        self.idx_count[ndx] -= count

        assert self.idx_count[ndx] >= 0

        if self.idx_count[ndx] == 0:
            del self.idx_count[ndx]
            del self.terms[term]
            del self.indexes[ndx]
            heapq.heappush(self.q, ndx)
            return ndx

    def by_index(self, ndx):
        return self.indexes.get(ndx)

    def __getstate__(self):
        return marshal.dumps(self.__dict__)

    def __setstate__(self, self_dict):
        self.__dict__.update(marshal.loads(self_dict))


class Docmap(object):
    def __init__(self):
        self.ndx_doc = []
        self.doc_ndx = {}

    def add(self, doc_id):
        try:
            return self.doc_ndx[doc_id]
        except KeyError:
            self.ndx_doc.append(doc_id)
            idx = len(self.ndx_doc) - 1
            self.doc_ndx[doc_id] = idx
            return idx

    def append(self, docmap):
        "Specially for macro cluster merging"
        offset = len(self.ndx_doc)
        self.ndx_doc += docmap.ndx_doc
        for docid, ndx in docmap.doc_ndx.iteritems():
            self.doc_ndx[docid] = ndx + offset

    def remove_docs(self, doc_ids):
        ndxes = [self.doc_ndx[docid] for docid in doc_ids]
        ndxes.sort(reverse=True)
        for ndx in ndxes:
            del self.ndx_doc[ndx]

        self.rebuild_doc_ndx()

    def rebuild_doc_ndx(self):
        self.doc_ndx = dict((docid, ndx) for ndx, docid in enumerate(self.ndx_doc))

    def __getitem__(self, docid):
        return self.doc_ndx[docid]

    def by_index(self, ndx):
        return self.ndx_doc[ndx]

    def __delitem__(self, docid):
        ndx = self.doc_ndx[docid]
        del self.ndx_doc[ndx]
        del self.doc_ndx[docid]
        self.rebuild_doc_ndx()

    def del_by_index(self, ndx):
        docid = self.ndx_doc[ndx]
        del self.ndx_doc[ndx]
        del self.doc_ndx[docid]
        del self.lengths[ndx]
        self.rebuild_doc_ndx()

    def __contains__(self, docid):
        return docid in self.doc_ndx

    def __len__(self):
        return len(self.ndx_doc)

    def __repr__(self):
        return repr(self.ndx_doc)

    def cluster_docs(self, cluster):
        res = []
        for doc in self.ndx_doc:
            clid, itemd = doc.split(':')
            if clid == cluster:
                res.append(doc)
        return res

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __iter__(self):
        return enumerate(self.ndx_doc)


class Doc(object):
    def __init__(self, id, proximity=.0, status='work', fulltext=0, source_rating=0.15, mtime=None, resource_id=None,
                 project_resources=None):
        self.id = id
        self.proximity = proximity
        self.status = status
        self.fulltext = fulltext
        self.source_rating = source_rating
        self.mtime = mtime or dt.datetime.now()
        self._resource_id = resource_id
        if project_resources is None:
            project_resources = lazyload.project_resources()
        self._main_resource = project_resources['main'].get(resource_id)
        self._profile_resource = project_resources['profile'].get(resource_id, set())

    def __hash__(self):
        return self.id

    def __repr__(self):
        return '{id}{main}{profile}'.format(
            id=self.id,
            main='({}+)'.format(self.main_resource) if self.main_resource else '',
            profile='({})'.format(','.join(map(str, self.profile_resource))) if self.profile_resource else ''
        )

    def __eq__(self, other):
        if isinstance(other, Doc):
            return (
                self.id == other.id and
                self.status == other.status and
                str(self.proximity)[:7] == str(other.proximity)[:7]
            )
        return self.id == other

    @property
    def main_resource(self):
        return getattr(self, '_main_resource', None)

    @property
    def profile_resource(self):
        return getattr(self, '_profile_resource', set())

    @property
    def resource_id(self):
        return getattr(self, '_resource_id', -1)

    @property
    def is_main_resource(self):
        return bool(self.main_resource)

    def as_dict(self):
        return dict(
            id=self.id,
            weight=self.proximity,
            resource_id=self.resource_id,
            is_main_resource=self.is_main_resource,
            fulltext=self.fulltext
        )


class Cluster(list):
    def __init__(self, id, status=''):
        super(Cluster, self).__init__()
        self.id = id
        self.status = status
        self.constraints = set()
        self.ctime = dt.datetime.now()
        self.manual_pitem = None
        self.manual_pos = False
        self.has_meta = False

    project_id = property()

    @project_id.getter
    def project_id(self):
        if not getattr(self, '_project_id', None):
            project_ids = projects.projects_set(self)
            self._project_id = len(project_ids) == 1 and next(iter(project_ids)) or None
        return self._project_id

    @project_id.setter
    def project_id(self, value):
        self._project_id = getattr(self, '_project_id', None) or value

    def __repr__(self):
        return '{id}{project_id}: {docs}'.format(
            id=self.id,
            project_id=self.project_id and '<{}>'.format(self.project_id) or '',
            docs=super(Cluster, self).__repr__()
        )

    def doc_ids(self):
        return [doc.id for doc in self]

    def sort(self):
        return list.sort(
            self,
            key=lambda d: (not bool(d.is_main_resource), not bool(d.profile_resource), d.fulltext != 1, -d.proximity)
        )


class Clusters(dict):

    def __init__(self):
        super(Clusters, self).__init__()
        # R2 matrices (or total matrices, in case of 1-level clusterization)
        self.clmap = Docmap()
        self.centroids = None
        self.docspace = None
        self.docmap = Docmap()
        self.dictionary = Dictionary()


class MacroCluster(Clusters):
    "Contains MicroClusters"
    def __init__(self, id, status=''):
        super(MacroCluster, self).__init__()
        self.id = id
        self.status = status
        self.docmap_r1 = Docmap()
        self.docspace_r1 = None
        self.fixed_centroid = False

    def doc_ids(self):
        ids = []
        for c in self.values():
            ids.extend(c.doc_ids())
        return ids
    
    def __repr__(self):
        return '[%s] %s' % (self.id, super(MacroCluster, self).__repr__())


class MacroClusters(collections.MutableMapping):

    def __init__(self):
        self.clmap_r1 = Docmap()
        self.centroids_r1 = None
        self.dictionary_r1 = Dictionary()
        self.dictionary_r2 = Dictionary()
        self.macro_clusters = dict()
        self.doc_cluster = dict()  # doc to cluster relationship

    def __getitem__(self, key):
        return self.macro_clusters[key]

    def __setitem__(self, clid, macro):
        self.macro_clusters[clid] = macro

    def __delitem__(self, key):
        # TODO: update doc_cluster
        del self.macro_clusters[key]

    def __len__(self):
        return len(self.macro_clusters)

    def __iter__(self):
        return iter(self.macro_clusters)

    def __contains__(self, item):
        return item in self.macro_clusters

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['doc_cluster'] = marshal.dumps(self_dict['doc_cluster'])
        return self_dict

    def __setstate__(self, self_dict):
        self_dict['doc_cluster'] = marshal.loads(self_dict['doc_cluster'])
        self.__dict__.update(self_dict)

    def loaded(self, clid):
        "Dummy method for tests."
        return self.macro_clusters[clid] is not None

    def preload(self, clids):
        "Dummy method for tests."
        pass
