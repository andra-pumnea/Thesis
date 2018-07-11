from __future__ import print_function

import argparse
import sys
import datetime
import time

from keras.callbacks import History, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

import vocab
import preprocessing
import dec_att
import esim
import gru
import namespace_utils
import numpy as np
import pandas as pd

# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'


def run(FLAGS):
    if FLAGS.task == "quora":
        # quora_full_dataset = FLAGS.full_dataset
        train_file = FLAGS.train_path
        dev_file = FLAGS.dev_path
        test_file = FLAGS.test_path
        embeddings = FLAGS.embeddings
        model = FLAGS.model
        maxlen = FLAGS.max_sent_length
        max_nb_words = FLAGS.max_sent_length

        vocab.prepare_vocab(train_file, embeddings)

        # Prepare datasets
        q1_train, q2_train, y_train, word_embedding_matrix = preprocessing.prepare_dataset(train_file,
                                                                                           maxlen,
                                                                                           max_nb_words, 1)
        q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file, maxlen, max_nb_words)
        q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file, maxlen, max_nb_words)

        if model == "dec_att":
            net = dec_att.create_model(word_embedding_matrix, maxlen)
        elif model == "esim":
            net = esim.create_model(word_embedding_matrix, maxlen)
        elif model == "gru":
            net = gru.create_model(word_embedding_matrix, maxlen)
        elif model == "bimpm":
            pass

        # Start training
        print("Starting training at", datetime.datetime.now())
        t0 = time.time()
        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
        history = net.fit([q1_train, q2_train],
                          y_train,
                          validation_data=([q1_dev, q2_dev], y_dev),
                          batch_size=FLAGS.batch_size,
                          nb_epoch=FLAGS.max_epochs,
                          shuffle=True,
                          callbacks=callbacks)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        max_val_acc, idx = get_best(history)
        test_loss, test_acc = evaluate_best_model(net, q1_test, q2_test, y_test)

        print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))
        print('loss = {0:.4f}, accuracy = {1:.4f}'.format(test_loss, test_acc))

        plot_acc_curve(history)

        # compute final accuracy on training and test sets
        # scores = net.evaluate([q1_test, q2_test], y_test)
        # print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))

        # evaluate_model(net, q1_train, q2_train, y_train)
        # print("%.2f%% (+/- %.2f%%)" % (mean, variance))


def get_best(history):
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    return max_val_acc, idx


def evaluate_best_model(model, q1_test, q2_test, y_test):
    model.load_weights(MODEL_WEIGHTS_FILE)
    loss, accuracy = model.evaluate([q1_test, q2_test], y_test, verbose=0)
    return loss, accuracy


def plot_acc_curve(history):
    acc = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                        'training': history.history['acc'],
                        'validation': history.history['val_acc']})
    ax = acc.iloc[:, :].plot(x='epoch', figsize={5, 8}, grid=True)
    ax.set_ylabel("accuracy")
    ax.set_ylim([0.0, 1.0]);


def evaluate_model(net, q1_train, q2_train, y_train):
    # define 10-fold cross validation test harness
    seed = 7
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    X = zip(q1_train, q2_train)

    for train, test in kfold.split(X, y_train):
        # # Start training
        # net.fit([q1_train, q2_train], y_train,
        #         validation_data=([q1_dev, q2_dev], y_dev),
        #         batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs, shuffle=True, )

        # # compute final accuracy on training and test sets
        # scores = net.evaluate([q1_test, q2_test], y_test)
        # print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))

        net.fit(X[train], y_train[train], epochs=2, batch_size=64, verbose=0)
        # evaluate the model
        scores = net.evaluate(X[test], y_train[test], verbose=0)
        print("%s: %.2f%%" % (net.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    return np.mean(cvscores), np.std(cvscores)


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
