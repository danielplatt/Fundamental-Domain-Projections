import tensorflow as tf


def get_he_network(inputSize = 12*15):
    inp = tf.keras.layers.Input(shape=(12,15,))
    prep = tf.keras.layers.Reshape((12*15,))(inp)
    h1 = tf.keras.layers.Dense(1000, activation='sigmoid')(prep)
    h2 = tf.keras.layers.Dense(100, activation='tanh')(h1)
    out = tf.keras.layers.Dense(1, activation='linear')(h2)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=[],
    )
    return model


if __name__ == '__main__':
    model = get_he_network()
    print(model.summary())
