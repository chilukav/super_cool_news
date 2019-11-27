# -*- coding: utf-8 -*-
"""
By classes I call our etalon clusterization, while clusters form clusterization, that we want to measure.
"""

import math
from itertools import chain

import numpy


def cluster_precision(klass, cluster):
    "Expects klass and cluster variables as set()"
    return float(len(klass & cluster)) / len(cluster)


def cluster_recall(klass, cluster):
    "Expects klass and cluster variables as set()"
    return float(len(klass & cluster)) / len(klass)


def fm(alpha, klass, cluster):
    "F-measure for one class and one cluster"
    p = cluster_precision(klass, cluster)
    r = cluster_recall(klass, cluster)
    if not (p and r):
        return 0.0
    return (1.0 + alpha) * r * p / (alpha * p + r)


def precision(klasses, clusters):
    avg_precision = 0.0
    total_weights = sum(map(lambda k: len(k), clusters))
    for cluster in clusters:
        likely_precision, likely_klass = max(map(lambda k: (cluster_precision(k, cluster), k), klasses))
        avg_precision += len(cluster) * likely_precision

    return avg_precision / total_weights


def recall(klasses, clusters):
    avg_recall = 0.0
    total_weights = sum(map(lambda k: len(k), klasses))
    for cluster in clusters:
        likely_recall, likely_klass = max(map(lambda k: (cluster_recall(k, cluster), k), klasses))
        avg_recall += len(likely_klass) * likely_recall

    return avg_recall / total_weights


def f_measure(alpha, total_docs_n, klasses, clusters):
    result = 0.0
    for klass in klasses:
        result += len(klass) * max([fm(alpha, klass, cluster) for cluster in clusters])
    return result / total_docs_n


def avg_f_measure(alpha, klasses, clusters):
    result = 0.0
    for cluster in clusters:
        result += max([fm(alpha, klass, cluster) for klass in klasses])

    return result / len(clusters)


def purity(total_docs_n, klasses, clusters):
    result = 0.0
    for cluster in clusters:
        result += len(cluster) * max([cluster_precision(klass, cluster) for klass in klasses])
    return result / total_docs_n


def entropy(total_docs_n, klasses, clusters):
    result = 0.0
    for cluster in clusters:
        mean = 0.0
        for klass in klasses:
            p = cluster_precision(klass, cluster)
            if not p: continue
            mean += p * math.log(p)
        result += mean * len(cluster)
    result /= total_docs_n

    return result * (-1.0 / math.log(len(clusters), 10))


def NMI(total_docs_n, klasses, clusters):
    """
    Normalized mutual information. It is a clustering accuracy measure
    that is tolerant to mismatches between number of clusters and classes.
    """
    result = 0.0
    for cluster in clusters:
        for klass in klasses:
            inters = len(cluster & klass)
            if not inters: continue
            inters = float(inters)
            result += inters * math.log((total_docs_n * inters) / (len(cluster) * len(klass)), 10)

    norm1 = 0.0
    for cluster in clusters:
        norm1 += len(cluster) * math.log(float(len(cluster)) / total_docs_n, 10)

    norm2 = 0.0
    for klass in klasses:
        norm2 += len(klass) * math.log(float(len(klass)) / total_docs_n, 10)

    return result / math.sqrt(norm1 * norm2)
