import collections
import functools

from cluster.clusterizer import log

logger = log.get_logger("projects")


def _log(s):
    logger.debug('[projects] %s', ' '.join(s.split()))


def projects_set(docs):
    if not isinstance(docs, collections.Iterable):
        docs = [docs]

    for d in docs:
        if d.is_main_resource:
            return {d.main_resource}

    projects = filter(bool, [d.profile_resource for d in docs])
    return projects and functools.reduce(lambda p1, p2: p1 & p2, projects) or set()


def main_resource(docs):
    if not isinstance(docs, collections.Iterable):
        docs = [docs]

    for doc in docs:
        if doc.is_main_resource:
            return doc


def log_result(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        result = fun(*args, **kwargs)
        _log('%s (args %s, kwargs %s): %s' % (fun.__name__, args, kwargs, result))
        return result
    return wrapper


@log_result
def main_check(obj1, obj2):
    main1, main2 = main_resource(obj1), main_resource(obj2)
    return (main1 is None or main2 is None) or (main1 == main2)


@log_result
def projects_check(obj1, obj2):
    def stored_projects_set(obj):
        project_id = getattr(obj, 'project_id', None)
        return project_id and {project_id} or projects_set(obj)

    projects1, projects2 = stored_projects_set(obj1), stored_projects_set(obj2)
    return (not projects1 or not projects2) or bool(projects1 & projects2)


def patch_proximities(docs, clids, clusters, proximities, new_docs=False):
    def check(doc, cluster):
        if doc.is_main_resource and not new_docs:
            return 1 if doc in cluster else 0

        if not main_check(doc, cluster) or not projects_check(doc, cluster):
            return 0

    for doc_ndx, doc in docs:
        if not projects_set(doc):
            continue

        for cl_ndx, clid in enumerate(clids):
            cluster = clusters[clid]
            proximity = check(doc, cluster)
            if proximity is not None:
                _log('Doc %s, cluster %s (%s), proximity %s' % (doc, clid, cluster, proximity))
                proximities[cl_ndx, doc_ndx] = proximity

    return proximities
