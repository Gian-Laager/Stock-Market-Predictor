import yfinance as yf
import tensorflow as tf
import numpy as np
import DataCollectors as dataCollectors

# TODO step function
# TODO reset function


class DateDict(dict):
    def __init__(self, iterable={}):
        super().__init__(iterable)

    def __getitem__(self, key):
        if type(key) != slice:
            return super().__getitem__(key)

        else:
            keys = list(super().keys())[list(super().keys()).index(key.start): list(super().keys()).index(key.stop)]
            retDict = DateDict()
            for k in keys:
                retDict[k] = super().__getitem__(k)
            
            return retDict

    def tolist(self):
        retList = []
        for k in list(super().keys()):
            retList.append(super().__getitem__(k))
        return retList

class Env:
    def __init__(self, stockName, stateSize=194, dataCollectorData=True):
        self.dfData = yf.download(
            tickers=stockName, period='60d', interval='2m')
        self.stateSize = stateSize
        self.dateIndex = self.stateSize
        self.actionSize = 2
        self.cAction = 0 # 0 = sell, 1 = buy, if it is the same two times then it means hold
        self.dates = self.extractDates()
        self.stockName = stockName
        self.dataCollector = False
        if dataCollectorData:
            self.dataCollector = dataCollectors.BackPropegationDataCollector(
                self.stockName, f'./Data/{self.stockName}-DQN.txt')
        self.dictData = self.createDictData() # the current date of the enviorment as index in the date list
        self.buyTime = None # as index in the dates list
        self.reward = 0
        self.cProfit = 0

    def getState(self):
        self.updateData()
        return tf.Variable(list(self.dictData[list(self.dictData.keys())[self.dateIndex - self.stateSize]:list(self.dictData.keys())[self.dateIndex]].values()))

    def getReward(self):
        self.updateReward()
        return self.reward +self.cProfit

    def extractDates(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def removeNanFromData(self, dictionary):
        keys = list(dictionary.keys())

        for key in keys:
            if True in tf.math.is_nan(dictionary[key]).numpy().flatten().tolist():
                if list(dictionary.keys()).index(key) == self.dateIndex:
                    self.dateIndex += 1
                del dictionary[key]
                np.delete(self.dates, self.dates.tolist().index(key))

        return dictionary

    def createDictData(self, dictionary=DateDict()):
        rawData = self.dfData.to_numpy()

        for i in range(len(self.dates)):
            if not self.dates[i] in dictionary.keys():
                dictionary[self.dates[i]] = tf.Variable(
                    np.average(rawData[i][:-1]))

        if self.dataCollector != False:
            collectorDict = self.dataCollector.getDict()
            for key in list(collectorDict.keys()):
                if not key in dictionary.keys():
                    dictionary[key] = collectorDict[key]
        dictionary = self.removeNanFromData(dictionary)

        return dictionary

    def updateData(self):
        # self.dfData = yf.download(
        #     tickers=self.stockName, period='60d', interval='2m')
        self.dates = self.extractDates()
        self.dictData = self.createDictData(self.dictData)

    def updateReward(self):
        if self.buyTime != None:
            self.cProfit = self.dictData[self.dates[self.buyTime]] - self.dictData[self.dates[self.dateIndex]]

        if self.cAction == 0 and self.buyTime != None:
            self.reward += self.cProfit 
            self.cProfit = 0
            self.buyTime = None

        elif self.cAction == 1 and self.buyTime == None:
            self.buyTime = self.dateIndex

    def step(self, act):
        self.dateIndex += 1
        self.updateData()
        self.cAction = act

        return self.getState(), self.getReward()

    def reset(self):
        self.updateData()
        self.reward = 0
        self.cProfit = 0
        self.dateIndex = self.stateSize
        self.buyTime = None
        self.cAction = 0