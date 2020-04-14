import yfinance as yf
import tensorflow as tf
import numpy as np

class DateDict(dict):
    def __init__(self, iterable={}):
        super().__init__(iterable)

    def __getitem__(self, key):
        if type(key) != slice:
            return super().__getitem__(key)

        else:
            keys = list(super.keys())[key]
            retDict = DateDict()
            for k in keys:
                retDict[k] = super().__getitem__(k)

    def tolist(self):
        retList = []
        for k in list(super().keys()):
            retList.append(super().__getitem__(k))
        return retList

class Env:
    def __init__(self, stockName, stateSize=194):
        self.dfData = yf.download(tickers=stockName, period='60d', interval='2m')
        self.stateSize = stateSize
        self.dates = self.extractDates()
        self.dictData = self.createDictData()
        self.dictDataKeyIndex = self.stateSize - 1 # the current date of the enviorment as index in the data dictonary
        self.buyTime = None
        self.reward = 0
        self.cProfit = 0
    
    def extractDates(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def ceateDictData(self):
        rawData = tf.Variable(self.dfData.to_numpy())

        dictionary = DateDict()
        for i in range(len(self.dates)):
            dictionary[self.dates[i]] = rawData[i]
 
        if self.dataCollector != False:
            collectorDict = self.dataCollector.getDict()
            for key in list(collectorDict.keys()):
                dictionary[key] = collectorDict[key]

        return dictionary

        