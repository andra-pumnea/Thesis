
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def tfidf_fit(X):
    tfidf = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
    tfidf.fit(X)

    # Taken as a consideration
    # if a word was never seen - it must be at least as infrequent
    # as any of the known words - so the default idf is the max of
    # known idf's
    max_idf = max(tfidf.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return word2weight


def tfidf_transform(X, embeddings, word2weight):
    return np.array([
        np.mean([embeddings[w] * word2weight[w] for w in words if w in embeddings] or
                [np.zeros(300)], axis=0)
        for words in X
    ])


def fit_transform(X, embeddings):
    word2weight = tfidf_fit(X)
    return tfidf_transform(X, embeddings, word2weight)

