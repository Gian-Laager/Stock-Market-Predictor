import tensorflow as tf

def createDenseModel(pretrainedModel=False, filePath=None):
    def loadModel(*args, **kwargs):
        if filePath == None:
            raise AttributeError('if you want to load a model you have to spesifie the file path')

        return tf.keras.models.load_model('./Models/DQNModel.h5')

    def buildModel(stateShape, actionSize):
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
            model.add(tf.keras.layers.Dense(8, activation=tf.keras.activations.sigmoid))

        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(actionSize, activation=tf.keras.activations.sigmoid))

        model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam())

        return model

    if pretrainedModel:
        return loadModel
    else:
        return buildModel