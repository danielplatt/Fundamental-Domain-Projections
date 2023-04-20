from tensorflow.keras.models import Model
from data.load_data import load_data
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, LeakyReLU, Dropout, ZeroPadding2D, BatchNormalization, Flatten, concatenate
from tensorflow.keras.optimizers   import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from util.soft_acc import soft_acc


def get_erbin_finotello_network():
    h11_deep_inception_model = scan_inception_model(input_shape=(12,15,),
                                                    model_name='h11_inception_same_conv_no_fc',
                                                    learning_rate=1.0e-3,
                                                    conv_layers=[32, 64, 32],
                                                    conv_padding='same',
                                                    conv_alpha=0.0,
                                                    fc_layers=[],
                                                    fc_alpha=0.0,
                                                    dropout=0.3,
                                                    full_dropout=0.0,
                                                    normalization=0.99,
                                                    last_relu=True,
                                                    l1_reg=1.0e-4,
                                                    l2_reg=1.0e-4
                                                    )
    return h11_deep_inception_model

def scan_inception_model(input_shape,
                         model_name='inception_model',
                         learning_rate=0.01,
                         conv_layers=[32],
                         conv_padding='same',
                         conv_alpha=0.0,
                         fc_layers=[],
                         fc_alpha=0.0,
                         dropout=0.2,
                         full_dropout=0.0,
                         normalization=0.99,
                         last_relu=True,
                         out_name='output',
                         l1_reg=0.0,
                         l2_reg=0.0
                         ):
    '''
    Build a CNN 'scan-inception' model: scan over rows and columns and merge.

    Required arguments:
        input_size:    the size of the input tensor.

    Optional arguments:
        model_name:    the name of the model,
        learning_rate: the learning rate of the gradient descent,
        conv_layers:   a list-like object with the no. of filters for each 'inception' modules,
        conv_padding:  the padding to use for the convolutional scans,
        conv_alpha:    the slope of the LeakyReLU activation (ReLU if 0.0) of the convolution layers,
        fc_layers:     a list-like object with the no. of units for each hidden dense layer,
        fc_alpha:      the slope of the LeakyReLU activation (ReLU if 0.0) of the FC network,
        dropout:       the dropout rate (do not use dropout if <= 0.0),
        full_dropout:  use this dropout rate after every layer (disabled if <= 0.0),
        normalization: the momentum of the batch normalization (do not use normalization if <= 0.0),
        last_relu:     whether to use ReLU activation in the output layer (force positive output),
        out_name:      the name of the output layer,
        l1_reg:        the L1 kernel regularization factor,
        l2_reg:        the L2 kernel regularization factor.

    Returns:
        the compiled model.
    '''

    # define the regularizer
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg)  # --------------------------------------------- regularizer

    # build the model
    I = Input(shape=input_shape, name=model_name + '_input')  # ------------------------------ input layer

    # reshape from 12*15 vector to (12,15) matrix
    # only change compared to original function
    I_reshaped = tf.keras.layers.Reshape((12, 15, 1))(I)

    if conv_padding == 'same':
        x = I_reshaped
    else:
        x = ZeroPadding2D(padding=((0, 3), (0, 0)), data_format='channels_last')(I_reshaped)

    # build convolutional layers
    for n in range(
            np.shape(conv_layers)[0]):  # --------------------------------------------- loop through the conv. layers
        a = Conv2D(filters=conv_layers[n],
                   kernel_size=(x.shape[1], 1),
                   padding=conv_padding,
                   kernel_regularizer=regularizer,
                   name=model_name + '_conv2d_rows_' + str(n)
                   )(
            x)  # -------------------------------------------------------------------- add conv. layer over rows
        a = LeakyReLU(alpha=conv_alpha,
                      name=model_name + '_conv2d_rows_' + str(n) + '_activation'
                      )(a)  # ----------------------------------------------------------------- add activation
        b = Conv2D(filters=conv_layers[n],
                   kernel_size=(1, x.shape[2]),
                   padding=conv_padding,
                   kernel_regularizer=regularizer,
                   name=model_name + '_conv2d_columns_' + str(n)
                   )(
            x)  # -------------------------------------------------------------------- add conv. layer over columns
        b = LeakyReLU(alpha=conv_alpha,
                      name=model_name + '_conv2d_columns_' + str(n) + '_activation'
                      )(b)  # ----------------------------------------------------------------- add activation

        x = concatenate([a, b],
                        name=model_name + '_concatenation_' + str(n)
                        ) if conv_padding == 'same' \
            else concatenate([a, tf.einsum('bij...->bji...', b)],  # ------------- swap columns and rows
                             name=model_name + '_concatenation_' + str(n)
                             )  # ------------------------------------------------ concatenate layers

        if normalization > 0.0:
            x = BatchNormalization(momentum=normalization,
                                   name=model_name + '_conv2d_' + str(n) + '_normalization'
                                   )(
                x)  # ---------------------------------------------------- add batch normalization (if requested)
        if full_dropout > 0.0:
            x = Dropout(rate=full_dropout,
                        name=model_name + '_conv2d_' + str(n) + '_full_dropout'
                        )(
                x)  # --------------------------------------------------------------- add dropout (if requested)

    # add dropout
    if dropout > 0.0 and full_dropout <= 0.0:
        x = Dropout(rate=dropout,
                    name=model_name + '_dropout'
                    )(
            x)  # ------------------------------------------------------------------- add dropout (if requested)

    # flatten the output
    x = Flatten(name=model_name + '_flatten')(x)  # ------------------------------------------ flatten the output

    # build FC network
    for n in range(np.shape(fc_layers)[0]):
        x = Dense(units=fc_layers[n],
                  kernel_regularizer=regularizer,
                  name=model_name + '_fc_' + str(n)
                  )(x)  # --------------------------------------------------------------------- add dense layers
        x = LeakyReLU(alpha=fc_alpha,
                      name=model_name + '_fc_' + str(n) + '_activation'
                      )(x)  # ----------------------------------------------------------------- add activation

        if normalization > 0.0:
            x = BatchNormalization(momentum=normalization,
                                   name=model_name + '_fc_' + str(n) + '_normalization'
                                   )(
                x)  # ---------------------------------------------------- add batch normalization (if requested)
        if full_dropout > 0.0:
            x = Dropout(rate=full_dropout,
                        name=model_name + '_fc_' + str(n) + '_full_dropout'
                        )(
                x)  # --------------------------------------------------------------- add dropout (if requested)

    if last_relu:  # ------------------------------------------------------------------------- output layer
        F = Dense(1, activation='relu', name=model_name + '_' + out_name)(x)
    else:
        F = Dense(1, name=model_name + '_' + out_name)(x)

    # define the model
    model = Model(inputs=I, outputs=F, name=model_name)

    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', soft_acc]
                  )

    # return the compiled model
    return model

def train_erbin_finotello(X_train, y_train, X_test, y_test):
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=400,
                               verbose=0,
                               restore_best_weights=True
                               ),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3,
                                   patience=150,
                                   verbose=0
                                   ),
                 ]
    # X_train, y_train = remove_outliers(X_train, y_train)
    model = get_erbin_finotello_network()
    history = model.fit(
        X_train, y_train,
        epochs=5000,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        batch_size=32
    )
    print(history.history['val_soft_acc'])
    return history.history['val_soft_acc'][-1]

def check_h11_range(y):
    outlier_counter = 0
    outliers = []
    for hodge in y:
        if hodge < 1 or hodge > 16:
            outlier_counter += 1
            outliers += [hodge]
    return outlier_counter, outliers

def remove_outliers(X_train, y_train):
    outlier_counter, outliers = check_h11_range(y_train)
    print(f'Training data contains {outlier_counter} outliers (i.e. h11<1 or h11>16) for the first Hodge number: {outliers}')
    print('Removing outliers...')
    outlier_indices = []
    for k in range(len(y_train)):
        if y_train[k] < 1 or y_train[k] > 16:
            outlier_indices += [k]
    print(outlier_indices)
    X_train = np.delete(X_train, outlier_indices, 0)
    y_train = np.delete(y_train, outlier_indices, 0)
    print('Outliers have been removed.')
    outlier_counter, outliers = check_h11_range(y_train)
    print(f'Training data now contains {outlier_counter} outliers (i.e. h11<1 or h11>16) for the first Hodge number: {outliers}')
    return X_train, y_train


if __name__ == '__main__':
    model = get_erbin_finotello_network()
    model.summary()
    data = load_data('', False)
    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1])[:, 0], test_size=0.5)
    print(f'Test Accuracy of Erbin-Finotello Neural Network after one run: {train_erbin_finotello(X_train, y_train, X_test, y_test)}')
