from __future__ import print_function

import argparse
import pickle
import sys
import datetime
import time

from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

import model_utils
import vocab
import preprocessing
import dec_att
import esim
import gru
import namespace_utils
import numpy as np
import pandas as pd
import matplotlib


# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score


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
    init_embeddings = 1

    word_index = vocab.prepare_vocab(train_file, embeddings)

    # Prepare datasets
    q1_train, q2_train, y_train, word_embedding_matrix = preprocessing.prepare_dataset(train_file,
                                                                                       maxlen,
                                                                                       max_nb_words,
                                                                                       experiment,
                                                                                       dataset,
                                                                                       init_embeddings)
    q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file, maxlen, max_nb_words, experiment, dataset)
    q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file, maxlen, max_nb_words, experiment, dataset)

    if dataset == 'snli':
        y_train = to_categorical(y_train, num_classes=None)
        y_dev = to_categorical(y_dev, num_classes=None)
        y_test = to_categorical(y_test, num_classes=None)

    if model == "dec_att":
        net = dec_att.create_model(word_embedding_matrix, maxlen)
    elif model == "esim":
        net = esim.create_model(word_embedding_matrix, maxlen)
    elif model == "gru":
        net = gru.create_model(word_embedding_matrix, maxlen)
    elif model == "bimpm":
        pass

    filepath = "models/weights.best.%s.%s.%s.hdf5" % (FLAGS.task, model, experiment)

    if mode == "load":
        print("Loading weights from %s" % filepath)
        net.load_weights(filepath)
        net.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                      metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    elif mode == "training":
        # Start training
        print("Starting training at", datetime.datetime.now())
        t0 = time.time()
        callbacks = [ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max'),
                     EarlyStopping(monitor='val_loss', patience=3)]
        history = net.fit([q1_train, q2_train],
                          y_train,
                          validation_data=([q1_dev, q2_dev], y_dev),
                          batch_size=FLAGS.batch_size,
                          nb_epoch=FLAGS.max_epochs,
                          shuffle=True,
                          callbacks=callbacks)

        pickle_file = "saved_history/history.%s.%s.%s.pickle" % (FLAGS.task, model, experiment)
        with open(pickle_file, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        max_val_acc, idx = get_best(history)
        print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))
    else:
        print("------------Unknown mode------------")

    test_loss, test_acc, test_f1 = evaluate_best_model(net, q1_test, q2_test, y_test, filepath)
    print('loss = {0:.4f}, accuracy = {1:.4f}, f1-score = {0:.4f}'.format(test_loss, test_acc * 100, test_f1))

    get_confusion_matrix(net, q1_test, q2_test, y_test)

    misclassified = get_misclassified_q(net, q1_test, q2_test, y_test, word_index)
    write_misclassified(misclassified)

    # plot_acc_curve(history)

    # compute final accuracy on training and test sets
    # scores = net.evaluate([q1_test, q2_test], y_test)
    # print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))

    # evaluate_model(net, q1_train, q2_train, y_train)
    # print("%.2f%% (+/- %.2f%%)" % (mean, variance))


def get_best(history):
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    return max_val_acc, idx


def evaluate_best_model(model, q1_test, q2_test, y_test, filepath):
    model.load_weights(filepath)
    scores = model.evaluate([q1_test, q2_test], y_test, verbose=0)
    loss = scores[1]
    accuracy = scores[2]
    f1_score = scores[3]
    print(scores)
    return loss, accuracy, f1_score


def get_confusion_matrix(model, q1_test, q2_test, y_test):
    y_pred = get_predictions(model, q1_test, q2_test)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification report:")
    target_names = ['non_duplicate', 'duplicate']
    print(classification_report(y_test, y_pred, target_names=target_names))


def get_misclassified_q(model, q1_test, q2_test, y_test, word_index):
    y_pred = get_predictions(model, q1_test, q2_test)

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


def get_predictions(model, q1_test, q2_test):
    y_pred = model.predict([q1_test, q2_test])
    y_pred = (y_pred > 0.5)
    y_pred = y_pred.flatten()
    y_pred = y_pred.astype(int)

    return y_pred


def write_misclassified(misclassified_q):
    output_file = "errors/misclassified.%s.%s.%s.tsv" % (FLAGS.task, FLAGS.model, FLAGS.experiment)
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
