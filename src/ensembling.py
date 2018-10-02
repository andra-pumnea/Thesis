from keras import Model, Input
from keras.layers import Average

import gru as gru
import dec_att as dec_att
import esim as esim


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


def ensemble(model_input, models):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(inputs=model_input, outputs=y, name='ensemble')

    return model
