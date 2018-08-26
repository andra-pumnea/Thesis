from __future__ import print_function
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import preprocessing
import model_utils
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


batch_size = 50
max_len = 40


def ElmoEmbedding(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size * [max_len])
    },
        signature="tokens",
        as_dict=True)["elmo"]


# https://www.kaggle.com/lamdang/dl-models
def create_model(pretrained_embedding, maxlen=30, embeddings='glove',
                 projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                 compare_dim=500, compare_dropout=0.2,
                 dense_dim=300, dense_dropout=0.2,
                 lr=1e-3, activation='elu'):
    # Based on: https://arxiv.org/abs/1606.01933

    if embeddings != 'elmo':
        q1 = Input(name='q1', shape=(maxlen,))
        q2 = Input(name='q2', shape=(maxlen,))
        # Embedding
        embedding = model_utils.create_pretrained_embedding(pretrained_embedding,
                                                            mask_zero=False)
        q1_embed = embedding(q1)
        q2_embed = embedding(q2)
    else:
        q1 = Input(shape=(maxlen,), dtype="string")
        q2 = Input(shape=(maxlen,), dtype="string")
        q1_embed = Lambda(ElmoEmbedding, output_shape=(maxlen, 1024))(q1)
        q2_embed = Lambda(ElmoEmbedding, output_shape=(maxlen, 1024))(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = model_utils.time_distributed(q1_embed, projection_layers)
    q2_encoded = model_utils.time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = model_utils.soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, model_utils.submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, model_utils.submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = model_utils.time_distributed(q1_combined, compare_layers)
    q2_compare = model_utils.time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = model_utils.apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = model_utils.apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    # out_ = Dense(3, activation='sigmoid')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy', model_utils.f1])

    return model


if __name__ == "__main__":
    train_file = 'Quora_question_pair_partition/train.tsv'
    dev_file = 'Quora_question_pair_partition/dev.tsv'
    test_file = 'Quora_question_pair_partition/test.tsv'
    # Prepare datasets
    q1_train, q2_train, y_train, word_embedding_matrix = preprocessing.prepare_dataset(train_file, 1)
    q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file)
    q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file)

    net = create_model(word_embedding_matrix)
    # Start training
    hist = net.fit([q1_train, q2_train], y_train,
                   validation_data=([q1_dev, q2_dev], y_dev),
                   batch_size=4, nb_epoch=20, shuffle=True, )

    # compute final accuracy on training and test sets
    scores = net.evaluate([q1_test, q2_test], y_test)
    print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))
