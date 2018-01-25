from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import Bidirectional, GRU
from keras.models import Model


def get_GRU_model(maxlen, max_features, embed_size, embed_weights, train_embed=False):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, trainable=train_embed, weights=embed_weights)(inp)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model
