import tensorflow as tf

def createDenseModel(stateShape, actionSize):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(64, input_shape=stateShape))

    for _ in range(16):
        for _ in range(16):
            model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.elu))

        model.add(tf.keras.layers.Dropout(0.5))

        for _ in range(16):
            model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.selu))

        model.add(tf.keras.layers.Dropout(0.25))

    for _ in range(8):
        model.add(tf.keras.layers.Dense(8, activation=tf.keras.activations.softplus))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(actionSize, activation=tf.keras.activations.sigmoid))

    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam())

    return model