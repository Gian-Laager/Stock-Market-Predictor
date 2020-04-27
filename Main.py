import tensorflow as tf
from DataGenerators import BackprobegationDataGenerator
from ModelCreators import createDenseModel
from DQN.Agents import Agent
from DQN.Env import Env
from Models import DeepLearingModel, DQNModel
from DQN.ModelCreators import createDenseModel as DQNDenseModelCreator

def main():
    STOCKNAME = 'AAPL'

    # model = DeepLearingModel(STOCKNAME, BackprobegationDataGenerator, createDenseModel)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./Models/checkpoints/DenseModel',
    #                                              save_weights_only=True,
    #                                              verbose=1)

    # model.fit(epochs=5, steps_per_epoch=10, callbacks=[cp_callback])
    # model.evaluate(steps=10)

    # model.save('./Models/DenseModel.h5')
    # model.plotLoss()

    model = DQNModel(STOCKNAME, Agent, Env, DQNDenseModelCreator(pretrainedModel=False, filePath='./Models/DQNModel2.h5'), './Models/DQNModel2.h5', batchSize=5)
    # model.env.plotData()
    model.fit(6)

if __name__ == "__main__":
    main()