from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from fuzzywuzzy import fuzz
import numpy as np
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stops = set(stopwords.words("english"))

def question_len(question1, question2):
    q1len = []
    q2len = []

    for q1, q2 in zip(question1, question2):
        q1, q2 = str(q1), str(q2)
        q1len.append([len(q1)])
        q2len.append([len(q2)])
    print("Created question character len feature")
    return np.array(q1len), np.array(q2len)


def question_words(question1, question2):
    q1words = []
    q2words = []

    for q1, q2 in zip(question1, question2):
        words_q1, words_q2 = str(q1).split(), str(q2).split()
        q1words.append([len(words_q1)])
        q2words.append([len(words_q2)])

    print("Created question word len feature")
    return np.array(q1words), np.array(q2words)


def word_match_share(question1, question2):
    word_overlap = []
    for q1, q2 in zip(question1, question2):
        q1words = {}
        q2words = {}
        for word in str(q1).lower().split():
            if word not in stops:
                q1words[word] = 1
        for word in str(q2).lower().split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            word_overlap.append([0])
        else:
            shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
            shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
            R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
            word_overlap.append([round(R, 2)])
    print("Created word_overlap feature")
    return np.array(word_overlap)


def tfidf_word_match_share(question1, question2):
    qs = question1 + question2
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(qs)
    feature_names = tfidf_vectorizer.get_feature_names()
    # dense = tfidf_matrix.todense()
    # word_index_dict = dict((j, i) for i, j in enumerate(feature_names))

    tf_idf = []
    for q1, q2 in zip(question1, question2):
        q1words = {}
        q2words = {}
        for word in str(q1).lower().split():
            if word not in stops:
                q1words[word] = 1
        for word in str(q2).lower().split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            tf_idf.append([0])
        else:
            q1_tfidf = tfidf_vectorizer.transform([" ".join(q1words.keys())])
            q2_tfidf = tfidf_vectorizer.transform([" ".join(q2words.keys())])
            inter = np.intersect1d(q1_tfidf.indices, q2_tfidf.indices)
            shared_weights = 0
            for word_index in inter:
                shared_weights += (q1_tfidf[0, word_index] + q2_tfidf[0, word_index])
            total_weights = q1_tfidf.sum() + q2_tfidf.sum()
            if np.sum(total_weights) == 0:
                tf_idf.append([0])
            else:
                score = np.sum(shared_weights) / np.sum(total_weights)
                tf_idf.append([round(score, 2)])
    print("Created tf_idf features feature")
    return np.array(tf_idf)


def compute_lda(question1, question2):
    seed = 1024
    lda = LatentDirichletAllocation(n_topics=10, doc_topic_prior=None,
                                    topic_word_prior=None, learning_method='batch',
                                    learning_decay=0.7, learning_offset=10.0, max_iter=10,
                                    batch_size=128, evaluate_every=-1, total_samples=1000000.0,
                                    perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100,
                                    n_jobs=1, verbose=0, random_state=seed)
    bow = CountVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=3, stop_words='english')
    vect_orig = make_pipeline(bow, lda)

    corpus = question1 + question2

    vect_orig.fit(corpus)

    lda = []
    for q1, q2 in zip(question1, question2):
        q1_lda = vect_orig.transform([q1])
        q2_lda = vect_orig.transform([q2])
        sim = cosine_similarity(q1_lda, q2_lda)
        lda.append([sim[0][0]])
    print("Created LDA feature feature")
    return np.array(lda)


def fw_qratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        qratio = fuzz.QRatio(str(q1), str(q2)) / 100
        fuzzy.append([qratio])
    print("Created fuzz qratio feature")
    return np.array(fuzzy)


def fw_wratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        WRatio = fuzz.WRatio(str(q1), str(q2)) / 100
        fuzzy.append([WRatio])
    print("Created fuzz wratio feature")
    return np.array(fuzzy)


def fw_partial_ratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        partial_ratio = fuzz.partial_ratio(str(q1), str(q2)) / 100
        fuzzy.append([partial_ratio])
    print("Created fuzz partial_ratio feature")
    return np.array(fuzzy)


def fw_partial_token_sort_ratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        partial_ratio = fuzz.partial_token_sort_ratio(str(q1), str(q2)) / 100
        fuzzy.append([partial_ratio])
    print("Created fuzz partial_token_sort_ratio feature")
    return np.array(fuzzy)


def fw_partial_token_set_ratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        partial_ratio = fuzz.partial_token_set_ratio(str(q1), str(q2)) / 100
        fuzzy.append([partial_ratio])
    print("Created fuzz partial_token_set_ratio feature")
    return np.array(fuzzy)


def fw_token_set_ratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        partial_ratio = fuzz.token_set_ratio(str(q1), str(q2)) / 100
        fuzzy.append([partial_ratio])
    print("Created fuzz token_set_ratio feature")
    return np.array(fuzzy)


def fw_token_sort_ratio(question1, question2):
    fuzzy = []
    for q1, q2 in zip(question1, question2):
        partial_ratio = fuzz.token_sort_ratio(str(q1), str(q2)) / 100
        fuzzy.append([partial_ratio])
    print("Created fuzz token_sort_ratio feature")
    return np.array(fuzzy)


def create_features(question1, question2):
    q1len, q2len = question_len(question1, question2)
    q1words, q2words = question_words(question1, question2)
    word_overlap = word_match_share(question1, question2)
    tfidf = tfidf_word_match_share(question1, question2)
    lda = compute_lda(question1, question2)
    qratio = fw_qratio(question1, question2)
    wratio = fw_wratio(question1, question2)
    partial_ratio = fw_partial_ratio(question1, question2)
    partial_token_sort_ratio = fw_partial_token_sort_ratio(question1, question2)
    partial_token_set_ratio = fw_partial_token_set_ratio(question1, question2)
    token_set_ratio = fw_token_set_ratio(question1, question2)
    token_sort_ratio = fw_token_sort_ratio(question1, question2)

    features = np.hstack([
        q1len,
        q2len,
        q1words,
        q2words,
        word_overlap,
        tfidf,
        lda,
        qratio,
        wratio,
        partial_ratio,
        partial_token_set_ratio,
        partial_token_sort_ratio,
        token_set_ratio,
        token_sort_ratio
    ])
    print('Feature vector size: %s' % features.shape[1])
    return features