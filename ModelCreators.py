import tensorflow as tf


def createDenseModel(inShape, outShape):
    model = tf.keras.Sequential()

    for _ in range(8):
        model.add(tf.keras.layers.Dense(
            32, activation=tf.keras.activations.relu))
    
    model.add(tf.keras.layers.Dropout(0.25))

    for _ in range(4):
        model.add(tf.keras.layers.Dense(
            8, activation=tf.keras.activations.linear))

    model.add(tf.keras.layers.Dense(6, activation=tf.keras.activations.linear))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='MSE')

    return model
