from __future__ import print_function

import argparse
import pickle
import sys
import datetime
import time

from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

import model_utils
import vocab
import preprocessing
import dec_att, dec_att_features
import esim, esim_features
import gru, gru_features
import namespace_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

# Setting the graph-level random seed.
tf.set_random_seed(89)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

# Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def run(FLAGS):
    # quora_full_dataset = FLAGS.full_dataset
    train_file = FLAGS.train_path
    dev_file = FLAGS.dev_path
    test_file = FLAGS.test_path
    embeddings = FLAGS.embeddings
    model = FLAGS.model
    mode = FLAGS.mode
    maxlen = FLAGS.max_sent_length
    max_nb_words = FLAGS.max_nb_words
    experiment = FLAGS.experiment
    dataset = FLAGS.task
    features = FLAGS.features
    init_embeddings = 1

    word_index = vocab.prepare_vocab(train_file, embeddings)

    # Prepare datasets
    q1_train, q2_train, y_train, word_embedding_matrix, features_train = preprocessing.prepare_dataset(train_file,
                                                                                                       maxlen,
                                                                                                       max_nb_words,
                                                                                                       experiment,
                                                                                                       dataset,
                                                                                                       features,
                                                                                                       init_embeddings)
    q1_dev, q2_dev, y_dev, features_dev = preprocessing.prepare_dataset(dev_file, maxlen, max_nb_words, experiment,
                                                                        dataset, features)
    q1_test, q2_test, y_test, features_test = preprocessing.prepare_dataset(test_file, maxlen, max_nb_words, experiment,
                                                                            dataset, features)

    if dataset == 'snli':
        y_train = to_categorical(y_train, num_classes=None)
        y_dev = to_categorical(y_dev, num_classes=None)
        y_test = to_categorical(y_test, num_classes=None)

    net = create_model(word_embedding_matrix)
    plot_file = "%s_plot.png" % FLAGS.model
    # plot_model(net, to_file=plot_file, show_shapes=True, show_layer_names=True)
    net.summary()

    filepath = "models/weights.best.%s.%s.%s.%s.hdf5" % (FLAGS.task, model, experiment, features)
    # filepath = "models/weights.best.quora.dec_att.training_full.hdf5"
    if mode == "load":
        print("Loading weights from %s" % filepath)
        net.load_weights(filepath)
        net.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                    metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    elif mode == "training":
        # Start training
        print("Starting training at", datetime.datetime.now())
        t0 = time.time()
        callbacks = get_callbacks(filepath)
        if not features_train and not features_dev:
            history = net.fit([q1_train, q2_train],
                              y_train,
                              validation_data=([q1_dev, q2_dev], y_dev),
                              batch_size=FLAGS.batch_size,
                              nb_epoch=FLAGS.max_epochs,
                              shuffle=True,
                              callbacks=callbacks)
        else:
            q1len_t, q2len_t, q1words_t, q2words_t = [x for x in features_train]
            q1len_d, q2len_d, q1words_d, q2words_d = [x for x in features_dev]
            history = net.fit([q1_train, q2_train, q1len_t, q2len_t, q1words_t, q2words_t],
                              y_train,
                              validation_data=(
                                  [q1_dev, q2_dev, q1len_d, q2len_d, q1words_d, q2words_d], y_dev),
                              batch_size=FLAGS.batch_size,
                              nb_epoch=FLAGS.max_epochs,
                              shuffle=True,
                              callbacks=callbacks)

        pickle_file = "saved_history/history.%s.%s.%s.%s.pickle" % (FLAGS.task, model, experiment, features)
        with open(pickle_file, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        max_val_acc, idx = get_best(history)
        print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))
    else:
        print("------------Unknown mode------------")

    get_confusion_matrix(net, q1_test, q2_test, y_test, features_test)

    misclassified = get_misclassified_q(net, q1_test, q2_test, y_test, word_index, features_test)
    write_misclassified(misclassified)

    cvscores = evaluate_model(word_embedding_matrix, q1_train, q2_train, y_train, features_train,
                              q1_test, q2_test, y_test, features_test, features)
    print("Finished running %s model on %s with %s" % (model, experiment, features))
    print_crossval(cvscores)
    print("Crossvalidation result: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    test_loss, test_acc, test_f1 = evaluate_best_model(net, q1_test, q2_test, y_test, filepath, features_test)
    print('Evaluation without crossval: loss = {0:.4f}, accuracy = {1:.4f}'.format(test_loss, test_acc * 100))


def create_model(word_embedding_matrix):
    model = FLAGS.model
    features = FLAGS.features
    maxlen = FLAGS.max_sent_length

    if model == "dec_att" and features == 'features':
        net = dec_att_features.create_model(word_embedding_matrix, maxlen)
    elif model == "dec_att" and features != 'features':
        net = dec_att.create_model(word_embedding_matrix, maxlen)
    elif model == "esim" and features == 'features':
        net = esim_features.create_model(word_embedding_matrix, maxlen)
    elif model == "esim" and features != 'features':
        net = esim.create_model(word_embedding_matrix, maxlen)
    elif model == "gru" and features == 'features':
        net = gru_features.create_model(word_embedding_matrix, maxlen)
    elif model == "gru" and features != 'features':
        net = gru.create_model(word_embedding_matrix, maxlen)
    return net


def print_crossval(cvscores):
    for idx, val in enumerate(cvscores):
        print('Fold%d: acc %.2f%%' % (idx, val))


def get_best(history):
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    return max_val_acc, idx


def evaluate_best_model(model, q1_test, q2_test, y_test, filepath, features):
    model.load_weights(filepath)

    if not features:
        scores = model.evaluate([q1_test, q2_test], y_test, verbose=0)
    else:
        q1len, q2len, q1words, q2words, word_overlap = [x for x in features]
        scores = model.evaluate([q1_test, q2_test, q1len, q2len, q1words, q2words, word_overlap], y_test, verbose=0)
    loss = scores[1]
    accuracy = scores[2]
    f1_score = scores[3]
    # print(scores)
    return loss, accuracy, f1_score


def get_confusion_matrix(model, q1_test, q2_test, y_test, features):
    y_pred = get_predictions(model, q1_test, q2_test, features)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification report:")
    target_names = ['non_duplicate', 'duplicate']
    print(classification_report(y_test, y_pred, target_names=target_names))


def get_misclassified_q(model, q1_test, q2_test, y_test, word_index, features):
    y_pred = get_predictions(model, q1_test, q2_test, features)

    misclassified_idx = np.where(y_test != y_pred)
    misclassified_idx = misclassified_idx[0].tolist()

    reverse_word_map = dict(map(reversed, word_index.items()))

    misclassified_q = []
    for i in misclassified_idx:
        pair = [q1_test[i], q2_test[i]]
        q_pair = []
        for idx, q in enumerate(pair):
            words = []
            for w in q:
                if w != 0:
                    word = reverse_word_map.get(w)
                    words.append(word)
            q_pair.append((' '.join(words).encode('utf-8').strip()))
        q_pair.append(y_test[i])
        q_pair.append(y_pred[i])
        misclassified_q.append(q_pair)
    return misclassified_q


def get_predictions(model, q1_test, q2_test, features):
    if not features:
        y_pred = model.predict([q1_test, q2_test])
    else:
        q1len, q2len, q1words, q2words, word_overlap = [x for x in features]
        y_pred = model.predict([q1_test, q2_test, q1len, q2len, q1words, q2words, word_overlap])
    y_pred = (y_pred > 0.5)
    y_pred = y_pred.flatten()
    y_pred = y_pred.astype(int)

    return y_pred


def write_misclassified(misclassified_q):
    output_file = "errors/misclassified.%s.%s.%s.%s.tsv" % (FLAGS.task, FLAGS.model, FLAGS.experiment, FLAGS.features)
    with open(output_file, 'w+') as f:
        for pair in misclassified_q:
            f.writelines(str(pair[0]) + '\t' + str(pair[1]) + '\t' + str(pair[2]) + '\t' + str(pair[3]) + '\n')
    print("Finished writing misclassified questions")


def plot_acc_curve(history):
    acc = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                        'training': history.history['acc'],
                        'validation': history.history['val_acc']})
    ax = acc.iloc[:, :].plot(x='epoch', figsize={5, 8}, grid=True)
    ax.set_ylabel("accuracy")
    ax.set_ylim([0.0, 1.0]);


def get_callbacks(filename):
    callbacks = [ModelCheckpoint(filename, monitor='val_loss', save_best_only=True, mode='min'),
                 EarlyStopping(monitor='val_loss', patience=5)]
    return callbacks


def evaluate_model(word_embedding_matrix, q1, q2, y, features_train, q1_dev, q2_dev, y_dev, features_dev, feat):
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvscores = []

    i = 1
    for train, test in kfold.split(q1, y):
        filepath = "cross_vall/fold%d.best.%s.%s.%s.%s.hdf5" % (i, FLAGS.task, FLAGS.model, FLAGS.experiment, feat)
        callbacks = get_callbacks(filepath)
        net = create_model(word_embedding_matrix)

        if not features_train:
            net.fit([q1[train], q2[train]], y[train],
                    validation_data=([q1[test], q2[test]], y[test]),
                    batch_size=FLAGS.batch_size,
                    nb_epoch=FLAGS.max_epochs,
                    shuffle=True,
                    callbacks=callbacks)

            # evaluate the model
            # net.load_weights(filepath)
            scores = net.evaluate([q1_dev, q2_dev], y_dev, verbose=0)
            print(scores)
        else:
            q1len, q2len, q1words, q2words, word_overlap = [x for x in features_train]
            q1len_d, q2len_d, q1words_d, q2words_d, word_overlap_d = [x for x in features_dev]
            net.fit([q1[train], q2[train], q1len[train], q2len[train], q1words[train], q2words[train],
                     word_overlap[train]], y[train],
                    validation_data=([q1[test], q2[test], q1len[test], q2len[test], q1words[test], q2words[test],
                                      word_overlap_d[test]], y[test]),
                    batch_size=FLAGS.batch_size,
                    nb_epoch=FLAGS.max_epochs,
                    shuffle=False,
                    callbacks=callbacks)
            # evaluate the model
            scores = net.evaluate([q1_dev, q2_dev, q1len_d, q2len_d, q1words_d, q2words_d,
                                   word_overlap_d], y_dev, verbose=0)
            print(scores)
        cvscores.append(scores[2] * 100)
        i += 1
    return cvscores


def enrich_options(options):
    key = "in_format"
    if key not in options.__dict__:
        options.__dict__["in_format"] = 'tsv'

    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_dataset', type=str, help='Path to full dataset for building vocab.')
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--embeddings', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--task', type=str, help='Dataset to be used.')
    parser.add_argument('--model', type=str, help='DL model run.')
    parser.add_argument('--mode', type=str, default='training', help='Mode to run the programme in.')

    parser.add_argument('--max_nb_words', type=int, default=80000, help='Maximum words in vocab.')
    parser.add_argument('--max_sent_length', type=int, default=30, help='Maximum words per sentence.')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension.')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances in each batch.')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum epochs for training.')

    parser.add_argument('--config_path', type=str, default='quora.config', help='Configuration file.')

    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        print('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()

    FLAGS = enrich_options(FLAGS)

    run(FLAGS)
