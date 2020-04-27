import tensorflow as tf
from matplotlib import pyplot as plt

class  ExtendedModel(tf.keras.Sequential):
    def __init__(self, layers=None, name=None):
        super().__init__(layers=layers, name=name)

    def plotLoss(self):
        plt.plot(self.history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        try:
            plt.plot(self.history.history['val_loss'])
        except KeyError:
            pass
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def fit(self, *args, **kwargs):
        self.history = super().fit(*args, **kwargs)


def createDenseModel(inShape, outShape):
    model = ExtendedModel()

    model.add(tf.keras.layers.Dense(
                64, activation=tf.keras.activations.linear, input_shape=(list(inShape)[-1],)))

    model.add(tf.keras.layers.Dropout(0.25))

    for _ in range(4):
        for _ in range(8):
            model.add(tf.keras.layers.Dense(
                32, activation=tf.keras.activations.linear))
            model.add(tf.keras.layers.Dropout(0.1))

        
        model.add(tf.keras.layers.Dropout(0.5))

        for _ in range(4):
            model.add(tf.keras.layers.Dense(
                8, activation=tf.keras.activations.linear))

        model.add(tf.keras.layers.Dropout(0.35))

    model.add(tf.keras.layers.Dense(list(outShape)[-1], activation=tf.keras.activations.linear))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss='MSE')

    return model
