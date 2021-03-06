from __future__ import print_function
import numpy as np
from os.path import expanduser, exists
import keras
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import csv
import pickle

from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import nltk
import re
import weight_embeddings
from gensim.models import FastText as fText

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
GLOVE_FILE = 'glove.840B.300d.txt'
FASTTEXT_FILE = '/home/andrada.pumnea/Data/Embeddings/wiki.en.vec'
# FASTTEXT_FILE = '/home/andrada.pumnea/Data/Embeddings/wiki-news-300d-1M-subword.vec'
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
    qid = []
    with open(file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(text_to_wordlist(row[1]))
            question2.append(text_to_wordlist(row[2]))
            is_duplicate.append(row[0])
            if len(row) == 4:
                qid.append(row[3])
            else:
                qid.append(0)
    print('Question pairs from %s: %d' % (file, len(question1)))

    return question1, question2, is_duplicate, qid


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
def get_embeddings(embeddings):
    embeddings_index = {}
    if embeddings == 'glove':
        file = KERAS_DATASETS_DIR + GLOVE_FILE
    else:
        file = FASTTEXT_FILE

    with open(file, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            if len(values) > 2:
                word = values[0]
                if embeddings != 'glove':
                    embedding = np.asarray(values[1:-1], dtype='float32')
                else:
                    embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))
    return embeddings_index


# Prepare word embedding matrix
def get_embedding_matrix(embeddings_index, word_index, max_nb_words):
    nb_words = max_nb_words
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    return word_embedding_matrix, nb_words


# Prepare word embedding matrix
def get_fasttext_embedding_matrix(word_index, max_nb_words):
    model = fText.load_fasttext_format(FASTTEXT_FILE)
    nb_words = max_nb_words
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    return word_embedding_matrix, nb_words


# Prepare word embedding matrix
def get_tfidf_embedding_matrix(embeddings_index, word_index, max_nb_words, word2weight):
    nb_words = min(max_nb_words, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_vector = embedding_vector * word2weight[word]
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


def init_embeddings(w_index, max_nb_words, task, experiment, embeddings):
    cache_filename = "embedding_matrix/%s.%s.%s.min.cache.npy" % (task, experiment, embeddings)
    # cache_filename = "embedding_matrix/snli.min.cache.npy"

    if exists(cache_filename):
        word_embedding_matrix = np.load(cache_filename)
        word_embedding_matrix = word_embedding_matrix[0]
    else:
        # Prepare embedding matrix to be used in Embedding Layer
        if embeddings == 'glove':
            embeddings_index = get_embeddings(embeddings)
            word_embedding_matrix = get_embedding_matrix(embeddings_index, w_index, max_nb_words)
        else:
            word_embedding_matrix = get_fasttext_embedding_matrix(w_index, max_nb_words)
        np.save(cache_filename, word_embedding_matrix)
    return word_embedding_matrix


def tokenize_text(data, max_len):
    questions = []
    for document in data:
        word_list = keras.preprocessing.text.text_to_word_sequence(document, lower=False)
        questions.append(word_list[0:max_len])
    return questions


def pad_tokens(data, max_len):
    new_data = []
    for seq in data:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_data.append(new_seq)
    return new_data


def prepare_glove(maxlen, question1, question2, is_duplicate):
    question1_word_sequences, question2_word_sequences, w_index = tokenize_data(question1, question2)
    q1_data, q2_data, labels = pad_sentences(question1_word_sequences, question2_word_sequences,
                                             is_duplicate, maxlen)
    return q1_data, q2_data, labels, w_index


#   elmo embeddings require strings not ints
def prepare_elmo(maxlen, question1, question2, is_duplicate):
    q1_tokens = tokenize_text(question1, maxlen)
    q2_tokens = tokenize_text(question2, maxlen)

    q1_data = pad_tokens(q1_tokens, maxlen)
    q2_data = pad_tokens(q2_tokens, maxlen)
    labels = np.array(is_duplicate, dtype=int)
    return np.array(q1_data), np.array(q2_data), labels


def prepare_dataset(filename, maxlen, max_nb_words, experiment, task, embeddings, train=0):
    question1, question2, is_duplicate, qid = read_dataset(filename)

    if embeddings != 'elmo':
        q1_data, q2_data, labels, w_index = prepare_glove(maxlen, question1, question2, is_duplicate)
    else:
        q1_data, q2_data, labels = prepare_elmo(maxlen, question1, question2, is_duplicate)

    X = np.stack((q1_data, q2_data), axis=1)
    y = labels

    Q1 = X[:, 0]
    Q2 = X[:, 1]
    q1_raw, q2_raw = prepare_sentence_enc(question1, question2)


    if train == 1:
        word_embedding_matrix = init_embeddings(w_index, max_nb_words, task, experiment, embeddings)
        return Q1, Q2, y, qid, q1_raw, q2_raw, word_embedding_matrix
    else:
        return Q1, Q2, y, qid, q1_raw, q2_raw


def sif_sentence_enc(question1, question2, embedding_size=300):
    qs = question1 + question2
    embed_index = get_embeddings('glove')
    matrix = weight_embeddings.sentence2vec(qs, embed_index, embedding_size)
    print(matrix.shape)

    with open('sif_sentences.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    q1_sent = matrix[:len(question1)]
    q2_sent = matrix[:len(question2)]
    print(q1_sent.shape)
    print(q2_sent.shape)


def get_existing_embedding_matrix(task='quora', experiment='training_full', embeddings='glove'):
    cache_filename = "embedding_matrix/%s.%s.%s.min.cache.npy" % (task, experiment, embeddings)
    # cache_filename = "embedding_matrix/snli.min.cache.npy"

    if exists(cache_filename):
        word_embedding_matrix = np.load(cache_filename)
        word_embedding_matrix = word_embedding_matrix[0]

    return word_embedding_matrix


#when generating the encoding, the embedding_matrix is needed
def tfidf_sentence_enc(question1, question2):
    embed_matrix = get_existing_embedding_matrix()
    qs = question1 + question2

    matrix = weight_embeddings.tfidf_fit_transform(qs, embed_matrix)
    print(matrix.shape)
    # with open('tfidf_sentences.pickle', 'wb') as handle:
    #     pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('tfidf_sentences.pickle', 'rb') as handle:
    #     matrix = pickle.load(handle)

    q1_sent = matrix[:len(question1)]
    q2_sent = matrix[:len(question2)]
    q1_sent = np.array(q1_sent, dtype='object')[:, np.newaxis]
    q2_sent = np.array(q2_sent, dtype='object')[:, np.newaxis]
    print(q1_sent.shape)
    print(q2_sent.shape)

    return q1_sent, q2_sent


def prepare_sentence_enc(question1, question2):
    q1_pre = []
    q2_pre = []
    for question1, question2 in zip(question1, question2):
        q1 = re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', question1)
        q2 = re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', question2)
        q1_pre.append(q1)
        q2_pre.append(q2)

    q1_raw = np.array(q1_pre, dtype='object')[:, np.newaxis]
    q2_raw = np.array(q2_pre, dtype='object')[:, np.newaxis]
    return q1_raw, q2_raw


def get_filename(path):
    split_file = path.rsplit('/', 1)
    return split_file[1]


# save the file everytime a new feature is added
# def handle_features(question1, question2, feat, task, experiment, file):
#     feature_filename = "feature_files/%s.%s.%s.features.npy" % (task, experiment, file)
#     features = np.array([])
#
#     if feat == 'features':
#         if exists(feature_filename):
#             features = np.load(feature_filename)
#         if not exists(feature_filename):
#             features = feature_module.create_features(question1, question2)
#             np.save(feature_filename, features)
#
#     return features

def squared_difference(vecs):
    x, y = vecs
    return K.sum(K.square(x - y), axis=-1, keepdims=True)


def abs_difference(vecs):
    x, y = vecs
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)


def multiplication(vecs):
    x, y = vecs
    print(K.shape(x), K.shape(y))
    return K.sum(x * y, axis=-1, keepdims=True)


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
