import tensorflow as tf


def get_bull_he_jejjala_mishra_network(inputSize = (12,15,)):
    inp = tf.keras.layers.Input(shape=inputSize)
    # prep = tf.keras.layers.Reshape((-1,))(inp)
    prep = tf.keras.layers.Reshape((12,15,1))(inp)

    conv_filter_numbers = [57, 56, 55, 43]
    c = [prep]
    for k in conv_filter_numbers:
        c += [
            tf.keras.layers.Conv2D(
                k, (3, 3), (1, 1), activation='relu'
            )(c[-1])
        ]
        c += [tf.keras.layers.Dropout(0.2072)(c[-1])]

    convolved = tf.keras.layers.Reshape((-1,))(c[-1])
    fc1 = tf.keras.layers.Dense(169, activation='relu')(convolved)
    fc1_dropped = tf.keras.layers.Dropout(0.2072)(fc1)
    fc2 = tf.keras.layers.Dense(491, activation='relu')(fc1_dropped)
    fc2_dropped = tf.keras.layers.Dropout(0.2072)(fc2)
    out = tf.keras.layers.Dense(20, activation='softmax')(fc2_dropped)
    # out = tf.keras.layers.Dense(20, activation='softmax')(prep)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':
    model = get_bull_he_jejjala_mishra_network()
    print(model.summary())
