from __future__ import print_function

import argparse
import pickle
import sys
import datetime
import time

from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K, Input
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Model

import preprocessing as preprocessing
import model_utils as model_utils
import vocab as vocab
import dec_att as dec_att
import esim as esim
import gru as gru
import infer_sent
import ensembling as ensembling
import namespace_utils as namespace_utils
import numpy as np
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
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


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
    sent_embed = FLAGS.sent_embed

    if embeddings == 'elmo':
        init_embeddings = 0
    else:
        init_embeddings = 1

    word_index = vocab.prepare_vocab(train_file, embeddings)

    # Prepare datasets
    if init_embeddings == 1:
        q1_train, q2_train, y_train, qid_train, raw1_train, raw2_train, q1_tfidf_train, q2_tfidf_train, word_embedding_matrix = preprocessing.prepare_dataset(
            train_file,
            maxlen,
            max_nb_words,
            experiment,
            dataset,
            embeddings,
            init_embeddings)
    else:
        q1_train, q2_train, y_train, qid_train, raw1_train, raw2_train, q1_tfidf_train, q2_tfidf_train = preprocessing.prepare_dataset(
            train_file,
            maxlen,
            max_nb_words,
            experiment,
            dataset,
            embeddings,
            init_embeddings)
        word_embedding_matrix = np.zeros(1)

    q1_dev, q2_dev, y_dev, qid_dev, raw1_dev, raw2_dev, q1_tfidf_dev, q2_tfidf_dev = preprocessing.prepare_dataset(
        dev_file, maxlen, max_nb_words,
        experiment,
        dataset, embeddings)
    q1_test, q2_test, y_test, qid_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test = preprocessing.prepare_dataset(
        test_file, maxlen,
        max_nb_words, experiment,
        dataset, embeddings)

    if dataset == 'snli':
        y_train = to_categorical(y_train, num_classes=None)
        y_dev = to_categorical(y_dev, num_classes=None)
        y_test = to_categorical(y_test, num_classes=None)

    net = create_model(word_embedding_matrix)
    net.summary()

    filepath = "models/weights.best.%s.%s.%s.%s.%s.hdf5" % (FLAGS.task, model, experiment, embeddings, sent_embed)
    if mode == "ensemble":
        print("Create ensemble of models")
    elif mode == "fine-tuning":
        model_file = "models/weights.best.quora.dec_att.training_full.glove.no_univ_sent.hdf5"
        print("Loading pre-trained model from %s:" % model_file)
        net.load_weights(model_file)
        net = freeze_layers(net)
        net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                    metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
        t0 = time.time()
        callbacks = get_callbacks(filepath)
        history = net.fit([q1_train, q2_train, raw1_train, raw2_train,q1_tfidf_train, q2_tfidf_train], y_train,
                          validation_data=([q1_dev, q2_dev, raw1_dev, raw2_dev,q1_tfidf_dev, q2_tfidf_dev], y_dev),
                          batch_size=FLAGS.batch_size,
                          nb_epoch=FLAGS.max_epochs,
                          shuffle=True,
                          callbacks=callbacks)

        pickle_file = "saved_history/history.%s.%s.%s.%s.%s.pickle" % (
            FLAGS.task, model, experiment, embeddings, sent_embed)
        with open(pickle_file, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
    elif mode == "transfer_learning":
        model_file = "models/weights.best.snli.esim.hdf5"
        print("Loading pre-trained model from %s:" % model_file)
        net.load_weights(model_file)
        net = replace_layer(net)
        net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                    metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
        t0 = time.time()
        callbacks = get_callbacks(filepath)
        history = net.fit([q1_train, q2_train, raw1_train, raw2_train,q1_tfidf_train, q2_tfidf_train], y_train,
                          validation_data=([q1_dev, q2_dev, raw1_dev, raw2_dev,q1_tfidf_dev, q2_tfidf_dev], y_dev),
                          batch_size=FLAGS.batch_size,
                          nb_epoch=FLAGS.max_epochs,
                          shuffle=True,
                          callbacks=callbacks)

        pickle_file = "saved_history/history.%s.%s.%s.%s.%s.pickle" % (
            FLAGS.task, model, experiment, embeddings, sent_embed)
        with open(pickle_file, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
    elif mode == "load":
        print("Loading weights from %s" % filepath)
        net.load_weights(filepath)
        net.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                    metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    elif mode == "training":
        # Start training
        print("Starting training at", datetime.datetime.now())
        t0 = time.time()
        callbacks = get_callbacks(filepath)
        history = net.fit([q1_train, q2_train, raw1_train, raw2_train,q1_tfidf_train, q2_tfidf_train], y_train,
                          validation_data=([q1_dev, q2_dev, raw1_dev, raw2_dev,q1_tfidf_dev, q2_tfidf_dev], y_dev),
                          batch_size=FLAGS.batch_size,
                          nb_epoch=FLAGS.max_epochs,
                          shuffle=True,
                          callbacks=callbacks)

        pickle_file = "saved_history/history.%s.%s.%s.%s.%s.pickle" % (
            FLAGS.task, model, experiment, embeddings, sent_embed)
        with open(pickle_file, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        max_val_acc, idx = get_best(history)
        print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))
    else:
        print("------------Unknown mode------------")

    get_confusion_matrix(net, q1_test, q2_test, y_test, raw1_test, raw2_test,q1_tfidf_test, q2_tfidf_test)

    predictions = find_prediction_probability(net, q1_test, q2_test, y_test, qid_test, raw1_test, raw2_test,q1_tfidf_test, q2_tfidf_test)
    write_predictions(predictions)

    # cvscores, loss_scores = evaluate_model(word_embedding_matrix, q1_train, q2_train, y_train
    #                                        q1_test, q2_test, y_test)
    print("Finished running %s model on %s with %s and %s" % (model, experiment, embeddings, sent_embed))
    # print_crossval(cvscores)
    # print("Crossvalidation accuracy result: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # print("Crossvalidation lostt result: %.2f (+/- %.2f)" % (np.mean(loss_scores), np.std(loss_scores)))

    if mode != "ensemble":
        test_loss, test_acc, test_f1 = evaluate_best_model(net, q1_test, q2_test, y_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test,
                                                           filepath)
    else:
        test_loss = evaluate_error(net, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test, y_test)
        test_acc = evaluate_accuracy(net, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test,y_test)
    print('Evaluation without crossval: loss = {0:.4f}, accuracy = {1:.4f}'.format(test_loss, test_acc * 100))

    with open("results.txt", "a") as myfile:
        myfile.write("Finished running %s model on %s with %s and %s" % (model, experiment, embeddings, sent_embed))
        myfile.write(
            'Evaluation without crossval: loss = {0:.4f}, accuracy = {1:.4f}'.format(test_loss, test_acc * 100))
        myfile.write('\n')


def create_model(word_embedding_matrix):
    model = FLAGS.model
    maxlen = FLAGS.max_sent_length
    embeddings = FLAGS.embeddings
    sent_embed = FLAGS.sent_embed

    if embeddings != 'elmo':
        q1 = Input(name='q1', shape=(maxlen,))
        q2 = Input(name='q2', shape=(maxlen,))
    else:
        q1 = Input(shape=(maxlen,), dtype="string")
        q2 = Input(shape=(maxlen,), dtype="string")

    q1_sent = Input(name='q1_sent', shape=(1,), dtype="string")
    q2_sent = Input(name='q2_sent', shape=(1,), dtype="string")

    q1_tfidf = Input(name='q1_tfidf', shape=(1, 300,))
    q2_tfidf = Input(name='q2_tfidf', shape=(1, 300,))

    model_input = [q1, q2, q1_sent, q2_sent, q1_tfidf, q2_tfidf]

    if model == "dec_att":
        net = dec_att.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    elif model == "esim":
        net = esim.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    elif model == "gru":
        net = gru.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    elif model == "infer_sent":
        net = infer_sent.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    elif model == "ensemble":
        models = ensemble_models(model_input, word_embedding_matrix)
        net = ensembling.ensemble(model_input, models)
    return net


def freeze_layers(model):
    for layer in model.layers[0:-1]:
        layer.trainable = False
    return model


def replace_layer(model):
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    output = model.get_layer('dropout_2').output
    # output = Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)
    new_model = Model(model.input, output)
    return new_model


def ensemble_models(model_input, word_embedding_matrix):
    maxlen = FLAGS.max_sent_length
    embeddings = FLAGS.embeddings
    sent_embed = FLAGS.sent_embed
    experiment = FLAGS.experiment

    decatt_file = "models/weights.best.%s.%s.%s.%s.%s.hdf5" % (
        FLAGS.task, 'dec_att', experiment, embeddings, sent_embed)
    esim_file = "models/weights.best.%s.%s.%s.%s.%s.hdf5" % (FLAGS.task, 'esim', experiment, embeddings, sent_embed)
    gru_file = "models/weights.best.%s.%s.%s.%s.%s.hdf5" % (FLAGS.task, 'gru', experiment, embeddings, sent_embed)
    models = ensembling.create_ensemble(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed,
                                        decatt_file, esim_file, gru_file)
    return models


def print_crossval(cvscores):
    for idx, val in enumerate(cvscores):
        print('Fold%d: acc %.2f%%' % (idx, val))


def get_best(history):
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    return max_val_acc, idx


def evaluate_best_model(model, q1_test, q2_test, y_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test,filepath):
    model.load_weights(filepath)

    scores = model.evaluate([q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test], y_test, verbose=0, batch_size=FLAGS.batch_size)
    loss = scores[1]
    accuracy = scores[2]
    f1_score = scores[3]
    # print(scores)
    return loss, accuracy, f1_score


def evaluate_error(model, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test, y_test):
    pred = model.predict([q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test], batch_size=FLAGS.batch_size)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error


def evaluate_accuracy(model, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test, y_test):
    y_pred = get_predictions(model, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test)
    score = accuracy_score(y_test, y_pred)
    return score


def get_confusion_matrix(model, q1_test, q2_test, y_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test):
    y_pred = get_predictions(model, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification report:")
    target_names = ['non_duplicate', 'duplicate']
    print(classification_report(y_test, y_pred, target_names=target_names))


def find_prediction_probability(model, q1_test, q2_test, y_test, qid_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test):
    y_pred = model.predict([q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test], batch_size=FLAGS.batch_size)

    question_pred = []
    for idx, i in enumerate(q1_test):
        pair = [qid_test[idx], y_test[idx], y_pred[idx]]
        question_pred.append(pair)
    return question_pred


def get_predictions(model, q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test):
    y_pred = model.predict([q1_test, q2_test, raw1_test, raw2_test, q1_tfidf_test, q2_tfidf_test], batch_size=FLAGS.batch_size)
    y_pred = (y_pred > 0.5)
    y_pred = y_pred.flatten()
    y_pred = y_pred.astype(int)

    return y_pred


def write_predictions(question_pred):
    output_file = "errors/predictions.%s.%s.%s.tsv" % (FLAGS.task, FLAGS.model, FLAGS.experiment)
    with open(output_file, 'w+') as f:
        for pair in question_pred:
            f.writelines(str(pair[0]) + '\t' + str(pair[1]) + '\t' + str(pair[2]) + '\n')
    print("Finished writing questions ids and their predictions")


def get_callbacks(filename):
    callbacks = [ModelCheckpoint(filename, monitor='val_loss', save_best_only=True, mode='min'),
                 EarlyStopping(monitor='val_loss', patience=3)]
    return callbacks


def evaluate_model(word_embedding_matrix, q1, q2, y, q1_dev, q2_dev, y_dev):
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvscores = []
    loss_scores = []

    i = 1
    for train, test in kfold.split(q1, y):
        filepath = "cross_vall/fold%d.best.%s.%s.%s.hdf5" % (i, FLAGS.task, FLAGS.model, FLAGS.experiment)
        callbacks = get_callbacks(filepath)
        net = create_model(word_embedding_matrix)

        net.fit([q1[train], q2[train]], y[train],
                validation_data=([q1[test], q2[test]], y[test]),
                batch_size=FLAGS.batch_size,
                nb_epoch=FLAGS.max_epochs,
                shuffle=False,
                callbacks=callbacks)

        # evaluate the model
        # net.load_weights(filepath)
        scores = net.evaluate([q1_dev, q2_dev], y_dev, verbose=0)
        print(scores)

        cvscores.append(scores[2] * 100)
        loss_scores.append(scores[1])
        i += 1
    return cvscores, loss_scores


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

    parser.add_argument('--config_path', type=str, default='configs/quora.config', help='Configuration file.')

    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        print('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()

    FLAGS = enrich_options(FLAGS)

    run(FLAGS)
