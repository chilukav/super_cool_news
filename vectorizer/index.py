import collections
import itertools
import math


class InvertedIndex:
    docs = {}
    posting_lists = collections.defaultdict(lambda: collections.defaultdict())

    def token_frequencies(self, ids=None):
        freq = {}
        texts = ids and self.get_texts(ids) or self.docs
        texts_count = len(self.docs)
        for text_id, text in texts.items():
            tfidf = {}
            tokens = text.get('title') + text.get('body')
            text_tokens_count = len(tokens)

            for token in tokens:
                token_texts = self.posting_lists.get(token, {})
                count = token_texts.get(text_id, 0)
                token_per_corpus = len(token_texts)

                tf = count / text_tokens_count
                token_freq = token_per_corpus > 1.e-10 and texts_count / token_per_corpus or 0.0
                idf = token_freq > 1.e-10 and math.log(token_freq) or 1.0
                tfidf[token] = tf * idf

            freq[text_id] = tfidf

        return freq

    def add(self, texts):
        tokens = collections.defaultdict(collections.Counter)
        for text in texts:
            text_id = text['id']
            title_tokens = text.get('title')
            body_tokens = text.get('body')

            self.docs[text_id] = text

            for token in itertools.chain(title_tokens, body_tokens):
                tokens[token][text_id] += 1

        for token, token_texts in tokens.items():
            for text_id, count in token_texts.items():
                self.posting_lists[token][text_id] = count

    def get(self, ids):
        texts = {id_: self.docs.get(id_) for id_ in ids if self.docs.get(id_) is not None}
        return texts
