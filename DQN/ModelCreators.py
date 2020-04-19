import tensorflow as tf

def createDenseModel(self, stateShape, actionSize):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(64, input_shape=stateShape[-1]))

    for _ in range(16):
        for _ in range(64):
            model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.elu))

        model.add(tf.keras.layers.Dropout(0.5))

        for _ in range(128):
            model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.selu))

        model.add(tf.keras.layers.Dropout(0.25))

    for _ in range(8):
        model.add(tf.keras.layers.Dense(8, activation=tf.keras.activations.softplus))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(actionSize, activation=tf.keras.activations.sigmoid))

    model.compile()