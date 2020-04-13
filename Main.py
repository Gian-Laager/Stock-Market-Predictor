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

    model.fit(dataset, epochs=10000, steps_per_epoch=10)

    model.save('./Models/DenseLayers6-4-20.h5')

if __name__ == "__main__":
    main()