import tensorflow as tf

class DQNModel:
    def __init__(self, stockName, Agent, Env, modelCreator, batchSize=64):
        self.env = Env(stockName, dataCollectorData=True)
        self.agent = Agent(self.env.stateSize, self.env.actionSize, modelCreator)
        self.batchSize = batchSize
        
    def fit(self, epochs):
        self.env.reset()
        for e in range(epochs):
            state = self.env.getState()
            action = self.agent.act(state)
            nextState, reward = self.env.step(action)
            self.agent.memorize(state, action, reward, nextState)

            if len(self.agent.memory) > self.batchSize:
                self.agent.fit(self.batchSize)

            print(f'Epoch: {e}, action: {action}, reward: {reward}, profit: {self.env.profit + self.env.cProfit}')

    def save(self, path):
        self.agent.save(path) 

