from keras import Model, Input
from keras.layers import Average

import gru as gru
import dec_att as dec_att
import esim as esim
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor


def create_ensemble(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed,
                    decatt_file, esim_file, gru_file):

    dec_att_model = dec_att.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    esim_model = esim.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)
    gru_model = gru.create_model(model_input, word_embedding_matrix, maxlen, embeddings, sent_embed)

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


def boost_model(create_model, model_input, y_train, y_test):
    ann_estimator = KerasRegressor(build_fn=create_model, epochs=50, batch_size=50, verbose=0)
    boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator)
    boosted_ann.fit(model_input, y_train)  # scale your training data
    boosted_ann.predict(y_test)
