import yfinance as yf
import tensorflow as tf
import numpy as np
import DataCollectors as dataCollectors
from matplotlib import pyplot as plt
import dataPreparation

class Env:
    def __init__(self, stockName, stateSize=194, dataCollectorData=True):
        self.stateSize = stateSize
        self.dateIndex = self.stateSize  # the current date of the enviorment as index in the date list
        self.actionSize = 2
        self.cAction = 0 # 0 = sell, 1 = buy, if it is the same two times then it means hold
        self.stockName = stockName
        self.dataCollector = False
        if dataCollectorData:
            self.dataCollector = dataCollectors.BackPropegationDataCollector(
                self.stockName, f'./Data/{self.stockName}-DQN.txt')
            self.dfData = self.dataCollector.dfData

        else:
            self.dfData = yf.download(
                tickers=stockName, period='60d', interval='2m')

        self.dictData = dataPreparation.createDictData(self.dfData, self.dataCollector)
        self.buyTime = None # as index in the dictData key list
        self.sellTime = self.dateIndex # as index in the dictData key list
        self.profit = 0
        self.reward = 0
        self.cProfit = 0

    def getState(self):
        self.updateData()
        return tf.Variable(self.dictData[self.dictData.keys()[self.dateIndex - self.stateSize]:self.dictData.keys()[self.dateIndex]].values())
    def getReward(self):
        self.updateReward()
        return self.reward

    def plotData(self):
        plotData = []
        dates = self.dictData.keys()
        for d in dates:
            plotData.append(self.dictData[d])
        plt.plot(plotData)
        plt.show()

    def updateData(self):
        self.dictData = dataPreparation.createDictData(self.dfData, self.dataCollector, self.dictData)

    def updateReward(self):
        if self.buyTime != None:
            self.cProfit = self.dictData[self.dictData.keys()[self.buyTime]] - self.dictData[self.dictData.keys()[self.dateIndex]]
            self.reward = self.dictData[self.dictData.keys()[self.buyTime]] - self.dictData[self.dictData.keys()[self.dateIndex]]

        elif self.cAction == 0 and self.sellTime != None:
            self.reward = -(self.dictData[self.dictData.keys()[self.sellTime]] - self.dictData[self.dictData.keys()[self.dateIndex]]) / 4

        if self.cAction == 0 and self.buyTime != None:
            self.profit += self.cProfit 
            self.cProfit = 0
            self.buyTime = None
            self.sellTime = self.dateIndex - 1
            self.reward = -(self.dictData[self.dictData.keys()[self.sellTime]] - self.dictData[self.dictData.keys()[self.dateIndex]]) / 4

        if self.cAction == 1 and self.buyTime == None:
            self.buyTime = self.dateIndex - 1
            self.sellTime = None
            self.reward = self.dictData[self.dictData.keys()[self.buyTime]] - self.dictData[self.dictData.keys()[self.dateIndex]]

    def step(self, act):
        self.dateIndex += 1
        self.updateData()
        self.cAction = act

        return self.getState(), self.getReward()

    def reset(self):
        self.updateData()
        self.reward
        self.profit = 0
        self.cProfit = 0
        self.dateIndex = self.stateSize
        self.buyTime = None
        self.cAction = 0