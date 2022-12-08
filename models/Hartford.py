from keras import layers, models, optimizers


def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # four parameters:
    # (1) Multiply every element of the matrix by a parameter
    # (2) Take the average of every row, which gives a 12x1 matrix. Write that 15 times next to each other to get a 12x15 matrix. Multiply the result by a parameter
    # (3) Same for columns
    # (4) Take the average of all matrix elements, which gives a 1x1 matrix. Repeat that number to get a 12x15 matrix. Multiply the result by a parameter
    # inp = layers.Reshape((12, 15, number_of_channels_in))(inp)

    # ---(1)---
    out1 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(inp)

    # ---(2)---
    out2 = layers.AveragePooling2D((1, 15), strides=(1, 1), padding='valid')(inp)
    repeated2 = [out2 for _ in range(15)]
    out2 = layers.Concatenate(axis=2)(repeated2)
    out2 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(out2)

    # ---(3)---
    out3 = layers.AveragePooling2D((12, 1), strides=(1, 1), padding='valid')(inp)
    repeated3 = [out3 for _ in range(12)]
    out3 = layers.Concatenate(axis=1)(repeated3)
    out3 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(out3)

    # ---(4)---
    out4 = layers.AveragePooling2D((12, 15), strides=(1, 1), padding='valid')(inp)
    repeated4 = [out4 for _ in range(12)]
    out4 = layers.Concatenate(axis=1)(repeated4)
    repeated4 = [out4 for _ in range(15)]
    out4 = layers.Concatenate(axis=2)(repeated4)
    out4 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=True, activation='relu')(out4)

    # return out1, out2, out3, out4
    return layers.Add()([out1,out2,out3,out4])

def get_hartford_network(pooling='sum'):
    number_of_channels = 100
    inp = layers.Input(shape=(12,15,1))
    inp_list = [inp for _ in range(number_of_channels)]
    inp_duplicated = layers.Concatenate(axis=3)(inp_list)
    e1 = equivariant_layer(inp_duplicated, number_of_channels, number_of_channels)
    # e1 = layers.Dropout(0.5)(e1)
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    # e2 = layers.Dropout(0.5)(e2)
    e3 = equivariant_layer(e2, number_of_channels, number_of_channels)
    # e3 = layers.Dropout(0.5)(e3)
    e4 = equivariant_layer(e3, number_of_channels, number_of_channels)
    # e4 = layers.Dropout(0.5)(e4)
    e5 = equivariant_layer(e4, number_of_channels, number_of_channels)
    # e5 = layers.Dropout(0.5)(e5)

    e6 = equivariant_layer(e5, number_of_channels, number_of_channels)
    # # e6 = layers.Dropout(0.5)(e6)
    # e7 = equivariant_layer(e6, number_of_channels, number_of_channels)
    # # e7 = layers.Dropout(0.5)(e7)
    # e8 = equivariant_layer(e7, number_of_channels, number_of_channels)
    # e9 = equivariant_layer(e8, number_of_channels, number_of_channels)

    if pooling=='sum':
        p1 = layers.AveragePooling2D((12, 15), strides=(1, 1), padding='valid')(e6)
    else:
        p1 = layers.MaxPooling2D((12, 15), strides=(1, 1), padding='valid')(e6)
    p2 = layers.Reshape((number_of_channels,))(p1)
    fc1 = layers.Dense(64, activation='relu')(p2)
    # fc1 = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=[],
    )
    return model


if __name__ == '__main__':
    model = get_hartford_network()
    model.summary()
