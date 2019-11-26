import datetime as dt
from os.path import abspath, dirname, join

# TODO UPDATE ALL VALUES HERE


DATA_PATH = '/data/clusterizer'
ITEM_ACTIVE_TIMEDELTA = dt.timedelta(days=3)
CLUSTER_ACTIVE_TIMEDELTA = dt.timedelta(days=1)  # TODO check this

# Resources
RESOURCES_DIR = '/home/rchannel/venv_scraper/data'
STOP_WORDS_PATH = join(RESOURCES_DIR, 'stop_words')
SPEC_DICT_PATH = join(RESOURCES_DIR, 'spec_dict')
BM25_IDF_PATH = join(RESOURCES_DIR, 'all_bm25idf.dat')
SPORT_CLUSTERS_PATH = join(RESOURCES_DIR, 'sport')
# End of resources

DUMPER_CLUSTERS_LIMIT = 100
DUMPER_OUTPUT_DIR = join(DATA_PATH, 'big_teplohod')
GOLDEN_DIR = join(DATA_PATH, 'golden_clusters')

TERMS_1LEVEL_PATH = join(DUMPER_OUTPUT_DIR, 'terms_1level.txt')
TERMS_2LEVEL_PATH = join(DUMPER_OUTPUT_DIR, 'terms_2level.txt')


IDX_OUTPUT = join(DATA_PATH, 'idx.pickle')
IDX_R1_OUTPUT = join(DATA_PATH, 'idx_r1.pickle')
IDX_R2_OUTPUT = join(DATA_PATH, 'idx_r2.pickle')

DOCMAP_OUTPUT = join(DATA_PATH, 'docmap.pickle')
DOCMAP_R1_OUTPUT = join(DATA_PATH, 'docmap_r1.pickle')
DOCMAP_R2_OUTPUT = join(DATA_PATH, 'docmap_r2.pickle')
ACTIVE_THRESHOLD = 0.15
FIRST_DOC_THRESHOLD = 0.2
CLUSTER_THRESHOLD = 0.2
CLUSTER_THRESHOLD2 = 0.1
DOC_FREQUENCY_MEASURE = 'BM25'
DOC_VECTOR_NORMALIZE = True
DOC_VECTOR_CUT = 20

TERMS_1LEVEL_PATH = join(DUMPER_OUTPUT_DIR, 'terms_1level_' + str(DOC_VECTOR_CUT) + '.txt')
TERMS_2LEVEL_PATH = join(DUMPER_OUTPUT_DIR, 'terms_2level_' + str(DOC_VECTOR_CUT) + '.txt')

BM25_K1 = 0.5
BM25_K2 = 1.5
BM25_GAMMA = 0.16
BM25_RO = 1.0
KMEANS_ITERATIONS = 5
KMEANS_CREATES_CLUSTER = True
MINIBOIL_ITERATIONS = 10
REMAINING_ITERATIONS = 10
KERNEL_THRESHOLD = 0.2
MIN_CLUSTER_SIZE = 1
GLUE_METHOD = 'MAIN'  # 'MAIN' or 'CENTROID'
GLUE_THRESHOLD = 0.2
GLUE_ITERATIONS = 10
SIMILAR_THRESHOLD = 0.5
GAUSSIAN_PROBABILITY = 0.1
CLUSTER_METHOD = 'kmeans'  # 'hier', 'kmeans'
HIERARCHICAL_THRESHOLD = 0.95


DOC_IMPORTANT_PART_LEN = 0.4   # Specifies the proportion of terms of document body
                            # that form important document part.(from the doc beggining)
                            # Like, if this parameter is 0.5, the the first half of the
                            # doc body is important, the rest is normal.

WEIGHTS = {
    'fulltext': 0.5,

    'doc_parts': {
        'title': 1.0,
        'important': 0.8,        # first, most significant, paragraph
        'normal': 0.8 * 0.5,     # main part of the news, less sigficant
        'remaining': 0.8 * 0.1,  # typically, last sentence, usually a crap
    },
}

MACRO_CLUSTER_THRESHOLD = 0.3
GLUE_THRESHOLD_MACRO = 0.4       # FIXME: glue threshould should be greater?!


MAIN_DOC_IS_CENTROID = False

TASKSET_TIMEOUT = 30 * 60

EXEC = 'celery'  # 'multiproc', 'direct', 'celery'

TREND_THRESHOLD_R1 = 0.2
TREND_THRESHOLD_R2 = 0.2

# Duplicator config
DUPLICATOR_SHINGLES_NUMBER = 5
DUPLICATOR_HASH_FNUMBER = 50
DUPLICATOR_HASH_SEED = 198
DUPLICATOR_SIMILARITY_THRESHOLD = 0.7
