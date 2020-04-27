import tensorflow as tf


class DQNModel:
    def __init__(self, stockName, Agent, Env, modelCreator, filePath, batchSize=64):
        self.filePath = filePath
        self.env = Env(stockName, dataCollectorData=True)
        self.agent = Agent(self.env.stateSize,
                           self.env.actionSize, modelCreator)
        self.batchSize = batchSize

    def fit(self, epochs):
        self.env.reset()
        if epochs >= 0:
            for e in range(1, epochs + 1):
                state = self.env.getState()
                action = self.agent.act(state)
                nextState, reward = self.env.step(action)
                self.agent.memorize(state, action, reward, nextState)

                if len(self.agent.memory) > self.batchSize:
                    self.agent.fit(self.batchSize)

                print(
                    f'Epoch: {e}, action: {action}, reward: {reward}, profit: {self.env.profit + self.env.cProfit}')
                self.agent.save(self.filePath)
        else:
            epoch = 1
            while True:
                state = self.env.getState()
                action = self.agent.act(state)
                nextState, reward = self.env.step(action)
                self.agent.memorize(state, action, reward, nextState)

                if len(self.agent.memory) > self.batchSize:
                    self.agent.fit(self.batchSize)

                print(
                    f'Epoch: {epoch}, action: {action}, reward: {reward}, profit: {self.env.profit + self.env.cProfit}')
                self.agent.save(self.filePath)
                epoch += 1

    def save(self, path):
        self.agent.save(path)


class DeepLearingModel:
    def __init__(self, stockName, dataGenarator, modelCreator, dataCollectorData=True):
        self.stockName = dataGenarator
        self.dataGenarator = dataGenarator(stockName, dataCollectorData)
        self.model = modelCreator(self.dataGenarator.sampleShape, self.dataGenarator.lableShape)
        self.dataset = tf.data.Dataset.from_generator(
            self.dataGenarator, (self.dataGenarator.sampleDtype, self.dataGenarator.lableDtype), (self.dataGenarator.sampleShape, self.dataGenarator.lableShape))
    
    def fit(self, *args, **kwargs):
        self.model.fit(self.dataset, *args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        self.model.evaluate(self.dataset, *args, **kwargs)
    
    def save(self, path):
        self.model.save(path)

    def plotLoss(self):
        self.model.plotLoss()
