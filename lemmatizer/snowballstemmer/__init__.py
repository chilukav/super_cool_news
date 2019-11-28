# Copyright (c) 2001, Dr Martin Porter
# Copyright (c) 2004,2005, Richard Boulton
# Copyright (c) 2013, Yoshiki Shibukawa
# Copyright (c) 2006,2007,2009,2010,2011,2014-2019, Olly Betts
# All rights reserved.


__all__ = ('language', 'stemmer')

from .english_stemmer import EnglishStemmer
from .russian_stemmer import RussianStemmer

_languages = {
    'english': EnglishStemmer,
    'russian': RussianStemmer,
}

try:
    import Stemmer
    cext_available = True
except ImportError:
    cext_available = False

def algorithms():
    if cext_available:
        return Stemmer.language()
    else:
        return list(_languages.key())

def stemmer(lang):
    if cext_available:
        return Stemmer.Stemmer(lang)
    if lang.lower() in _languages:
        return _languages[lang.lower()]()
    else:
        raise KeyError("Stemming algorithm '%s' not found" % lang)
