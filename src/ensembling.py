from keras import Model, Input
from keras.layers import Average

import src.gru as gru
import src.dec_att as dec_att
import src.esim as esim


def create_ensemble(word_embedding_matrix, maxlen, embeddings, sent_embed,
                    decatt_file, esim_file, gru_file):

    dec_att_model = dec_att.create_model(word_embedding_matrix, maxlen, embeddings, sent_embed)
    esim_model = esim.create_model(word_embedding_matrix, maxlen, embeddings, sent_embed)
    gru_model = gru.create_model(word_embedding_matrix, maxlen, embeddings, sent_embed)

    dec_att_model.load_weights(decatt_file)
    esim_model.load_weights(esim_file)
    gru_model.load_weights(gru_file)

    models = [dec_att_model, esim_model, gru_model]
    return models


def ensemble(models, maxlen, embeddings):
    if embeddings != 'elmo':
        q1 = Input(name='q1', shape=(maxlen,))
        q2 = Input(name='q2', shape=(maxlen,))
    else:
        q1 = Input(shape=(maxlen,), dtype="string")
        q2 = Input(shape=(maxlen,), dtype="string")

    q1_sent = Input(name='q1_sent', shape=(1,), dtype="string")
    q2_sent = Input(name='q2_sent', shape=(1,), dtype="string")

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(inputs=[q1, q2, q1_sent, q2_sent], outputs=y, name='ensemble')

    return model
