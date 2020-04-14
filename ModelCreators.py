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
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def fit(self, *args, **kwargs):
        self.history = super().fit(*args, **kwargs)


def createDenseModel(inShape, outShape):
    model = ExtendedModel()

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
