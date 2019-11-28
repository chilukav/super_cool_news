import snowballstemmer


def lemmatize(token, lang='russian'):
    stemmer = snowballstemmer.stemmer(lang)
    return stemmer.stemWord(token)
