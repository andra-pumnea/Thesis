from collections import defaultdict
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize


def tfidf_fit(X):
    tfidf = TfidfVectorizer(analyzer="word", stop_words='english', binary=False, ngram_range=(1,1),  tokenizer=word_tokenize)
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


def compute_word_probs(Xc):
    # compute word probabilities from corpus
    freqs = np.sum(Xc, axis=0).astype("float")
    probs = freqs / np.sum(freqs)
    return probs


def prepare_vocab(X, nb_words):
    # VOCAB_SIZE = X.shape[1]
    counter = CountVectorizer(stop_words="english",
                              max_features=nb_words)
    Xc = counter.fit_transform(X).todense().astype("float")

    sent_lens = np.sum(Xc, axis=1).astype("float")
    sent_lens[sent_lens == 0] = 1e-14
    print(sent_lens.shape)


def compute_svd(Xs):
    # compute 1st principal component
    svd = TruncatedSVD(n_components=1, n_iter=20, random_state=0)
    svd.fit(Xs)
    pc = svd.components_
    print(pc.shape, svd.explained_variance_ratio_)


def sif_transform(X, embeddings, nb_words):
    # from paper
    ALPHA = 1e-3
    Xc, sent_lens = prepare_vocab(X, nb_words)
    probs = compute_word_probs(Xc)

    # compute multiplier ALPHA / (ALPHA + probs)
    coeff = ALPHA / (ALPHA + probs)

    # compute weighted counts
    Xw = np.multiply(Xc, coeff)

    # convert to SIF embeddings
    Xs = np.divide(np.dot(Xw, embeddings[0]), sent_lens)
    print(Xc.shape, coeff.shape, Xs.shape, embeddings[0].shape)

    pc = compute_svd(Xs)
    Xr = Xs - Xs.dot(pc.T).dot(pc)
    print(Xr.shape)
    return Xr

