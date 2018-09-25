from __future__ import print_function

import model_utils
import preprocessing
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Lambda, Dense, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import History


history = History()
n_hidden = 250


def create_model(word_embedding_matrix, maxlen=30, embeddings='glove', sent_embed='univ_sent', lr=1e-5):
    # The visible layer
    if embeddings != 'elmo':
        question1 = Input(shape=(maxlen,))
        question2 = Input(shape=(maxlen,))

        in_dim, out_dim = word_embedding_matrix.shape
        embedding_layer = Embedding(in_dim,
                                    out_dim,
                                    weights=[word_embedding_matrix],
                                    input_length=maxlen,
                                    trainable=True)
        # Embedded version of the inputs
        encoded_q1 = embedding_layer(question1)
        encoded_q2 = embedding_layer(question2)
    else:
        question1 = Input(shape=(maxlen,), dtype="string")
        question2 = Input(shape=(maxlen,), dtype="string")
        encoded_q1 = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(question1)
        encoded_q2 = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(question2)

    q1_sent = Input(name='q1_sent', shape=(1,), dtype="string")
    q2_sent = Input(name='q2_sent', shape=(1,), dtype="string")
    q1_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q1_sent)
    q2_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q2_sent)
    sent1_dense = Dense(256, activation='relu')(q1_embed_sent)
    sent2_dense = Dense(256, activation='relu')(q2_embed_sent)
    distance_sent = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [sent1_dense, sent2_dense])

    # Since this is a siamese network, both sides share the same GRU
    shared_layer = GRU(n_hidden, kernel_initializer='glorot_uniform',
                       bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.0001))

    output_q1 = shared_layer(encoded_q1)
    output_q2 = shared_layer(encoded_q2)

    squared_diff = Lambda(preprocessing.squared_difference)([output_q1, output_q2])
    mult = Lambda(preprocessing.multiplication)([output_q1, output_q2])

    if sent_embed == 'univ_sent':
        merged = concatenate([output_q1, output_q2, squared_diff, mult, distance_sent])
    else:
        merged = concatenate([output_q1, output_q2, squared_diff, mult])

    merged = Dense(300, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(300, activation='relu')(merged)
    merged = BatchNormalization()(merged)

    output = Dense(1, activation='softmax', kernel_regularizer=regularizers.l2(0.0001),
                   bias_regularizer=regularizers.l2(0.0001))(merged)

    # Pack it all up into a model
    net = Model([question1, question2, q1_sent, q2_sent], [output])

    net.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    return net


# net.summary()
def run_gru():
    train_file = 'Quora_question_pair_partition/train.tsv'
    dev_file = 'Quora_question_pair_partition/dev.tsv'
    test_file = 'Quora_question_pair_partition/test.tsv'

    # Prepare datasets
    q1_train, q2_train, y_train, emb_matrix = preprocessing.prepare_dataset(train_file, 1)
    q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file)
    q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file)

    net = create_model(emb_matrix)
    # Start training
    hist = net.fit([q1_train, q2_train], y_train,
                   validation_data=([q1_dev, q2_dev], y_dev),
                   batch_size=4, nb_epoch=20, shuffle=True, )

    # compute final accuracy on training and test sets
    scores = net.evaluate([q1_test, q2_test], y_test)
    print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))

    # print(hist.history['acc'])
    # print(hist.history['val_acc'])
