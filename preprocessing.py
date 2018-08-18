from __future__ import print_function
import numpy as np
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import csv
import pickle
from nltk.corpus import stopwords
import nltk
import re

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
GLOVE_FILE = 'glove.840B.300d.txt'
EMBEDDING_DIM = 300

nltk.download('stopwords')
stops = set(stopwords.words("english"))

def text_to_wordlist(text, remove_stopwords=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    # Return a list of words
    return (text)


def read_dataset(file):
    print("Processing quora question pairs file")
    question1 = []
    question2 = []
    is_duplicate = []
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(text_to_wordlist(row[1]))
            question2.append(text_to_wordlist(row[2]))
            is_duplicate.append(row[0])
    print('Question pairs from %s: %d' % (file, len(question1)))

    return question1, question2, is_duplicate


def tokenize_data(question1, question2):
    questions = question1 + question2

    # load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    word_index = tokenizer.word_index
    print("Words in index: %d" % len(word_index))

    return question1_word_sequences, question2_word_sequences, word_index


# Create embedding index
def get_embeddings():
    embeddings_index = {}
    with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))
    return embeddings_index


# Prepare word embedding matrix
def get_embedding_matrix(embeddings_index, word_index, max_nb_words):
    nb_words = min(max_nb_words, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    return word_embedding_matrix, nb_words


# Prepare training data tensors
def pad_sentences(question1_word_sequences, question2_word_sequences, is_duplicate, maxlen):
    q1_data = pad_sequences(question1_word_sequences, maxlen=maxlen)
    q2_data = pad_sequences(question2_word_sequences, maxlen=maxlen)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)
    return q1_data, q2_data, labels


def init_embeddings(w_index, max_nb_words, task, experiment):
    cache_filename = "%s.%s.min.cache.npy" % (task, experiment)
    #cache_filename = "quora.training_full.min.cache.npy"

    if exists(cache_filename):
        word_embedding_matrix = np.load(cache_filename)
        word_embedding_matrix = word_embedding_matrix[0]
    else:
        # Prepare embedding matrix to be used in Embedding Layer
        embeddings_index = get_embeddings()
        word_embedding_matrix = get_embedding_matrix(embeddings_index, w_index, max_nb_words)
        np.save(cache_filename, word_embedding_matrix)
    return word_embedding_matrix


def prepare_dataset(filename, maxlen, max_nb_words, experiment, task, feat,train=0):
    question1, question2, is_duplicate = read_dataset(filename)
    question1_word_sequences, question2_word_sequences, w_index = tokenize_data(question1, question2)
    q1_data, q2_data, labels = pad_sentences(question1_word_sequences, question2_word_sequences,
                                             is_duplicate, maxlen)

    features = []
    if feat == 'features':
        features = create_features(question1, question2)

    X = np.stack((q1_data, q2_data), axis=1)
    y = labels

    Q1 = X[:, 0]
    Q2 = X[:, 1]

    if train == 1:
        word_embedding_matrix = init_embeddings(w_index, max_nb_words, task, experiment)
        return Q1, Q2, y, word_embedding_matrix, features
    else:
        return Q1, Q2, y, features


def question_len(question1, question2):
    q1len = []
    q2len = []

    for q1, q2 in zip(question1, question2):
        q1, q2 = str(q1), str(q2)
        q1len.append(len(q1))
        q2len.append(len(q2))
    return np.array(q1len), np.array(q2len)


def question_words(question1, question2):
    q1words = []
    q2words = []

    for q1, q2 in zip(question1, question2):
        words_q1, words_q2 = str(q1).split(), str(q2).split()
        q1words.append(len(words_q1))
        q2words.append(len(words_q2))

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
            word_overlap.append(0)
        else:
            shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
            shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
            R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
            word_overlap.append(round(R, 2))
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
            tf_idf.append(0)
        else:
            q1_tfidf = tfidf_vectorizer.transform([" ".join(q1words.keys())])
            q2_tfidf = tfidf_vectorizer.transform([" ".join(q2words.keys())])
            inter = np.intersect1d(q1_tfidf.indices, q2_tfidf.indices)
            shared_weights = 0
            for word_index in inter:
                shared_weights += (q1_tfidf[0, word_index] + q2_tfidf[0, word_index])
            total_weights = q1_tfidf.sum() + q2_tfidf.sum()
            if np.sum(total_weights) == 0:
                tf_idf.append(0)
            else:
                score = np.sum(shared_weights) / np.sum(total_weights)
                tf_idf.append(round(score, 2))
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
        lda.append(sim[0][0])
    return np.array(lda)


def create_features(question1, question2):
    q1len, q2len = question_len(question1, question2)
    q1words, q2words = question_words(question1, question2)
    word_overlap = word_match_share(question1, question2)
    tfidf = tfidf_word_match_share(question1, question2)
    lda = compute_lda(question1, question2)
    return [q1len, q2len, q1words, q2words, word_overlap, tfidf, lda]


def euclidean_distance(vecs):
    x, y = vecs
    return K.sqrt(K.sum(K.square(x - y), axis=-1, keepdims=True))


def cosine_distance(vecs):
    x, y = vecs
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def exponent_neg_manhattan_distance(vecs):
    x, y = vecs
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))


def get_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def compute_accuracy(predictions, labels):
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))


