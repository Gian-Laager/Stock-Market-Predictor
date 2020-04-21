import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone
from DataGenerators import BackprobegationDataGenerator
from ModelCreators import createDenseModel
from DQN.Agents import Agent
from DQN.Env import Env
from DQN.Model import DQNModel
from DQN.ModelCreators import createDenseModel

def main():
    STOCKNAME = 'AAPL'

    # dataGenarator = BackprobegationDataGenerator(STOCKNAME)

    # dataset = tf.data.Dataset.from_generator(
    #     dataGenarator, (dataGenarator.sampleDtype, dataGenarator.lableDtype), (dataGenarator.sampleShape, dataGenarator.lableShape))

    # model = createDenseModel(dataGenarator.sampleShape,
    #                          dataGenarator.lableShape)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./Models/checkpoints/DenseModel',
    #                                              save_weights_only=True,
    #                                              verbose=1)

    # model.fit(dataset, epochs=1000, steps_per_epoch=10, callbacks=[cp_callback])
    # model.evaluate(dataset, steps=10)

    # model.save('./Models/DenseModel.h5')
    # model.plotLoss()

    model = DQNModel(STOCKNAME, Agent, Env, createDenseModel)

    model.fit(300)

    model.save('./Models/DQNModel.h5')

if __name__ == "__main__":
    main()