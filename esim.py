from __future__ import print_function
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import preprocessing
import model_utils


# https://www.kaggle.com/lamdang/dl-models
def create_model(pretrained_embedding,
                 maxlen=30,
                 embeddings = 'glove',
                 lstm_dim=300,
                 dense_dim=300,
                 dense_dropout=0.5):

    # Based on arXiv:1609.06038
    if embeddings != 'elmo':
        q1 = Input(name='q1', shape=(maxlen,))
        q2 = Input(name='q2', shape=(maxlen,))
        # Embedding
        embedding = model_utils.create_pretrained_embedding(pretrained_embedding,
                                                            mask_zero=False)
        bn = BatchNormalization(axis=2)
        q1_embed = bn(embedding(q1))
        q2_embed = bn(embedding(q2))

    else:
        q1 = Input(shape=(maxlen,), dtype="string")
        q2 = Input(shape=(maxlen,), dtype="string")

        bn = BatchNormalization(axis=2)
        q1_embed = bn(Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(q1))
        q2_embed = bn(Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(q2))

    q1_sent = Input(name='q1_sent', shape=(1,), dtype="string")
    q2_sent = Input(name='q2_sent', shape=(1,), dtype="string")
    q1_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q1_sent)
    q2_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q2_sent)
    sent1_dense = Dense(256, activation='relu')(q1_embed_sent)
    sent2_dense = Dense(256, activation='relu')(q2_embed_sent)
    distance = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [sent1_dense, sent2_dense])

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = model_utils.soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, model_utils.submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, model_utils.submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = model_utils.apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = model_utils.apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep, distance])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    # out_ = Dense(3, activation='sigmoid')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2, q1_sent, q2_sent], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    return model


if __name__ == "__main__":
    train_file = 'Quora_question_pair_partition/train.tsv'
    dev_file = 'Quora_question_pair_partition/dev.tsv'
    test_file = 'Quora_question_pair_partition/test.tsv'

    # Prepare datasets
    q1_train, q2_train, y_train, emb_matrix = preprocessing.prepare_dataset(train_file, 1)
    q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file)
    q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file)

    net = create_model(emb_matrix)
    # Start training
    # Start training
    hist = net.fit([q1_train, q2_train], y_train,
                   validation_data=([q1_dev, q2_dev], y_dev),
                   batch_size=4, nb_epoch=20, shuffle=True, )

    # compute final accuracy on training and test sets
    scores = net.evaluate([q1_test, q2_test], y_test)
    print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))
