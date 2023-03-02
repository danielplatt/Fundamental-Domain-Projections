import numpy as np
import tensorflow as tf
from data.load_data import load_data
from sklearn.model_selection import train_test_split

from util.soft_acc import soft_acc


def get_he_network():
    inp = tf.keras.layers.Input(shape=(12,15,))
    prep = tf.keras.layers.Reshape((12*15,))(inp)
    h1 = tf.keras.layers.Dense(1000, activation='sigmoid')(prep)
    h2 = tf.keras.layers.Dense(100, activation='tanh')(h1)
    out = tf.keras.layers.Dense(1, activation='linear')(h2)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=[soft_acc],
    )
    return model

def train_he_network(X_train, y_train, X_test, y_test):
    model = get_he_network()
    history = model.fit(
        X_train, y_train,
        epochs=500,
        validation_data=(X_test, y_test),
    )
    return history.history['val_soft_acc'][-1]


if __name__ == '__main__':
    model = get_he_network()
    print(model.summary())
    data = load_data('', False)
    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1])[:,0], test_size=0.5)
    print(f'Test Accuracy of He Neural Network after one run: {train_he_network(X_train, y_train, X_test, y_test)}')
