import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone
from DataGenerators import BackprobegationDataGenerator
from ModelCreators import createDenseModel

def main():
    STOCKNAME = 'AAPL'

    dataGenarator = BackprobegationDataGenerator(STOCKNAME)

    dataset = tf.data.Dataset.from_generator(
        dataGenarator, (dataGenarator.sampleDtype, dataGenarator.lableDtype), (dataGenarator.sampleShape, dataGenarator.lableShape))

    model = createDenseModel(dataGenarator.sampleShape,
                             dataGenarator.lableShape)

    model.fit(dataset, epochs=10, steps_per_epoch=10, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss')])
    model.evaluate(dataset, steps=3)

    model.save('./Models/DenseModel.h5')
    model.plotLoss()

if __name__ == "__main__":
    main()