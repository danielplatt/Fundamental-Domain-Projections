import tensorflow as tf
from data.load_data import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np


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

def train_bull_he_jejjala_mishra_network(X_train, y_train, X_test, y_test):
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=100,
                               verbose=0,
                               restore_best_weights=True
                               ),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3,
                                   patience=5,
                                   verbose=0
                                   ),
                 ]

    model = get_bull_he_jejjala_mishra_network()
    history = model.fit(
        X_train, y_train,
        epochs=2000,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        batch_size=1
    )
    return history.history['val_soft_acc'][-1]


if __name__ == '__main__':
    model = get_bull_he_jejjala_mishra_network()
    model.summary()
    data = load_data('', False)
    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1])[:, 0], test_size=0.5)
    print(f'Test Accuracy of Erbin-Finotello Neural Network after one run: {train_bull_he_jejjala_mishra_network(X_train, y_train, X_test, y_test)}')
