from __future__ import print_function
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import preprocessing as preprocessing
import model_utils as model_utils

# https://www.kaggle.com/lamdang/dl-models
def create_model(model_input, pretrained_embedding,
                 maxlen=30,
                 embeddings = 'glove',
                 sent_embed='univ_sent',
                 lstm_dim=300,
                 dense_dim=300,
                 dense_dropout=0.5):

    # Based on arXiv:1609.06038
    if embeddings != 'elmo':
        # Embedding
        embedding = model_utils.create_pretrained_embedding(pretrained_embedding,
                                                            mask_zero=False)
        bn = BatchNormalization(axis=2)
        q1_embed = bn(embedding(model_input[0]))
        q2_embed = bn(embedding(model_input[1]))

    else:
        bn = BatchNormalization(axis=2)
        q1_embed = bn(Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(model_input[0]))
        q2_embed = bn(Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(model_input[1]))

    q1_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(model_input[2])
    q2_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(model_input[3])
    sent1_dense = Dense(256, activation='relu')(q1_embed_sent)
    sent2_dense = Dense(256, activation='relu')(q2_embed_sent)
    distance = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [sent1_dense, sent2_dense])

    lstm = LSTM(100)

    lstm_out_q1 = lstm(model_input[4])
    lstm_out_q2 = lstm(model_input[5])
    distance2 = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [lstm_out_q1, lstm_out_q2])


    # q1_embed = GaussianNoise(0.01)(q1_embed)
    # q2_embed = GaussianNoise(0.01)(q2_embed)

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
    if sent_embed == 'univ_sent':
        merged = Concatenate()([q1_rep, q2_rep, distance])
    elif sent_embed == 'tfidf':
        merged = Concatenate()([q1_rep, q2_rep, distance2])
    else:
        merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    # out_ = Dense(3, activation='sigmoid')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=model_input, outputs=out_)
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
