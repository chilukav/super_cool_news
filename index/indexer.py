import collections
import itertools
import hashlib
import math

import msgpack
import redis

from django.conf import settings
from scraper_common import db
from scraper_common.db import models
from scraper_common.index import tokenize


class Yndex(object):
    # Yet another eventually consistent inverted index
    CORPUS_KEY = 'yndx:{index_name}:corpus'
    DOC_KEY = 'yndx:{index_name}:doc:{doc_id}'
    WORD_KEY = 'yndx:{index_name}:word:{word_hash}'

    def __init__(self, index_name='common'):
        self.index_name = index_name
        self.master = redis.Redis(**settings.SCRAPER_INDEX_REDIS['master'])
        self.slave = redis.Redis(**settings.SCRAPER_INDEX_REDIS['slave'])
        self.corpus_key = self.CORPUS_KEY.format(index_name=self.index_name)

    def _word_key(self, word):
        word_hash = hashlib.md5(word.encode('utf-8')).hexdigest()
        return self.WORD_KEY.format(index_name=self.index_name, word_hash=word_hash)

    def add_doc(self, doc):
        self.add_docs([doc])

    def add_docs(self, docs):
        words = collections.defaultdict(collections.Counter)
        pipe = self.master.pipeline()
        for doc in docs:
            doc_id = doc['id']
            title_tokens = doc.get('title_tokens')
            body_tokens = doc.get('body_tokens')
            if title_tokens is None or body_tokens is None:
                title_tokens = list(tokenize(doc['title'], homonyms=False))
                body_tokens = list(tokenize(doc['body'], homonyms=False))
                doc['title_tokens'] = title_tokens
                doc['body_tokens'] = body_tokens

            for homonyms in itertools.chain(title_tokens, body_tokens):
                word = homonyms[0]
                words[word][doc_id] += 1

            doc_key = self.DOC_KEY.format(index_name=self.index_name, doc_id=doc_id)
            data = msgpack.packb(dict(id=doc_id, title_tokens=title_tokens, body_tokens=body_tokens))
            pipe.set(doc_key, data)
            pipe.expire(doc_key, 4 * 24 * 60 * 60)
            pipe.sadd(self.corpus_key, doc['id'])

        pipe.execute()

        for word, word_docs in words.items():
            word_key = self._word_key(word)
            for doc_id, count in word_docs.items():
                pipe.hset(word_key, doc_id, count)

            pipe.execute()

        return docs

    def get_doc(self, doc_id):
        docs = self.get_docs([doc_id])

        return docs and docs[0] or None

    def get_docs(self, doc_ids):
        docs = []
        missing = []
        for doc_id in doc_ids:
            doc_key = self.DOC_KEY.format(index_name=self.index_name, doc_id=doc_id)
            if not self.slave.exists(doc_key):
                missing.append(doc_id)
            else:
                data = self.slave.get(doc_key)
                doc = msgpack.unpackb(data, encoding='utf-8')
                docs.append(doc)

        if missing:
            with db.session(master=True) as session:
                items = session.query(
                    models.Item.id,
                    models.Item.title,
                    models.Item.body
                ).filter(models.Item.id.in_(missing))

                to_add = [dict(id=item.id, title=item.title, body=item.body) for item in items]
                self.add_docs(to_add)
                docs.extend(to_add)

        return docs

    def get_vectors(self, doc_ids):
        vectors = []
        docs = self.get_docs(doc_ids)
        corpora_size = self.slave.scard(self.corpus_key)
        for doc in docs:
            tfidf = {}
            tokens = doc.get('title_tokens') + doc.get('body_tokens')
            doc_words_count = len(tokens)

            for homonyms in tokens:
                word = homonyms[0]
                word_key = self._word_key(word)
                count = int(self.slave.hget(word_key, doc['id']) or 0.0)
                word_per_corpus = self.slave.hlen(word_key) or 0.0

                tf = float(count) / doc_words_count
                word_freq = word_per_corpus > 1.e-10 and float(corpora_size) / word_per_corpus or 0.0
                idf = word_freq > 1.e-10 and math.log(word_freq) or 1.0
                tfidf[word] = tf * idf

            vectors.append(tfidf)

        return vectors

    def vectorize_docs(self, docs):
        words = collections.defaultdict(collections.Counter)
        tokenized_docs = collections.defaultdict(list)
        for doc in docs:
            for homonyms in itertools.chain(doc['title_tokens'], doc['body_tokens']):
                word = homonyms[0]
                words[word][doc['id']] += 1
                tokenized_docs[doc['id']].append(word)

        # use idfs from index if possible
        idfs = {}
        corpora_size = self.master.scard(self.corpus_key) or len(tokenized_docs)
        for word, word_docs in words.items():
            if word not in idfs:
                word_key = self._word_key(word)
                word_per_corpus = self.master.hlen(word_key) or len(words[word])
                word_freq = word_per_corpus > 1.e-10 and float(corpora_size) / word_per_corpus or 0.0
                idf = word_freq > 1.e-10 and math.log(word_freq) or 1.0
                idfs[word] = idf

        vectors = {}
        space = collections.defaultdict(float)
        for doc_id, tokens in tokenized_docs.items():
            vector = {}
            doc_words_count = len(tokens)
            for word in tokens:
                word_count = words[word][doc_id]
                tf = float(word_count) / doc_words_count

                vector[word] = tf * idfs.get(word, 1.0)
                space[word] = max(tf * idfs.get(word, 1.0), space.get(word, 0.0))

            vectors[doc_id] = vector

        return vectors, space

    def remove_docs(self, doc_ids):
        if not doc_ids:
            return 0, 0, 0

        with self.master.pipeline(transaction=True) as pipe:
            words = collections.defaultdict(list)
            doc_keys = []
            for doc_id in doc_ids:
                doc_key = self.DOC_KEY.format(index_name=self.index_name, doc_id=doc_id)
                data = self.master.get(doc_key)
                if not data:
                    continue

                doc_keys.append(doc_key)
                doc = msgpack.unpackb(data, encoding='utf-8')
                for homonyms in itertools.chain(doc['title_tokens'], doc['body_tokens']):
                    word = homonyms[0]
                    words[word].append(doc['id'])

            for word, word_docs in words.items():
                word_key = self._word_key(word)
                pipe.hdel(word_key, *word_docs)

            if doc_keys:
                pipe.delete(*doc_keys)

            pipe.srem(self.corpus_key, *doc_ids)

            pipe.execute()

        return len(doc_ids), len(doc_keys), len(words)
