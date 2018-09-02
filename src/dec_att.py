from __future__ import print_function
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import preprocessing
import model_utils


# https://www.kaggle.com/lamdang/dl-models
def create_model(pretrained_embedding, maxlen=30, embeddings='glove', sent_embed='univ_sent',
                 projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                 compare_dim=500, compare_dropout=0.2,
                 dense_dim=300, dense_dropout=0.2,
                 lr=1e-3, activation='elu'):
    # Based on: https://arxiv.org/abs/1606.01933

    if embeddings != 'elmo' and embeddings != 'univ_sent':
        # Embedding

        q1 = Input(name='q1', shape=(maxlen,))
        q2 = Input(name='q2', shape=(maxlen,))
        embedding = model_utils.create_pretrained_embedding(pretrained_embedding,
                                                            mask_zero=False)
        q1_embed = embedding(q1)
        q2_embed = embedding(q2)
    else:
        q1 = Input(shape=(maxlen,), dtype="string")
        q2 = Input(shape=(maxlen,), dtype="string")
        q1_embed = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(q1)
        q2_embed = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(q2)

    q1_sent = Input(name='q1_sent', shape=(1,), dtype="string")
    q2_sent = Input(name='q2_sent', shape=(1,), dtype="string")
    q1_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q1_sent)
    q2_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(q2_sent)
    sent1_dense = Dense(256, activation='relu')(q1_embed_sent)
    sent2_dense = Dense(256, activation='relu')(q2_embed_sent)
    distance = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [sent1_dense, sent2_dense])

    q1_embed = GaussianNoise(0.01)(q1_embed)
    q2_embed = GaussianNoise(0.01)(q2_embed)

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
    merged = Concatenate()([q1_rep, q2_rep, distance])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    # out_ = Dense(3, activation='sigmoid')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2, q1_sent, q2_sent], outputs=out_)
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
