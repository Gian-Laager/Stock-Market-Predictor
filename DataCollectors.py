import yfinance as yf
import numpy as np
import tensorflow as tf
import numpy
from threading import Thread


def addThread(*funcs):
    for func in funcs:
        thread = Thread(target=func)
        thread.start()
        thread.join()


class TensroDecoder:
    def __init__(self, tensor):
        self.tensor = tensor

    def extractDtype(self, tensor):
        rawDtype = str(tensor.dtype)

        dtype = 'tf.' + rawDtype.split("'")[1]
        return dtype

    def __repr__(self):
        strValue = str(self.tensor.numpy().tolist())
        dtype = self.extractDtype(self.tensor)
        return f'tf.Variable({strValue}, dtype={dtype}, shape={self.tensor.shape})'


class BackPropegationDataCollector:
    def __init__(self, stockName, period='60d', interval='2m'):
        self.filePath = f'./Data/{stockName}-BackPropegationData.txt'
        self.stockName = stockName
        self.interval = interval
        self.period = period
        self.dfData, self.dictData, self.strDict = None, None, None
        addThread(self.updateFile())

    def extractDates(self, dataFrame):
        dates = dataFrame.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def convertDictDataToString(self, dictData):
        strDict = dict()
        for key in list(dictData.keys()):
            strDict[key] = TensroDecoder(dictData[key])

        return str(strDict)

    def convertDataFrameToDict(self, dataFrame):
        dates = self.extractDates(dataFrame)
        data = tf.Variable(dataFrame.to_numpy())

        dictData = dict()
        strDict = dict()
        for i in range(len(dates)):
            dictData[dates[i]] = data[i]

        return dictData

    def updateDictData(self):
        self.dfData = yf.download(
            tickers=self.stockName, period=self.period, interval=self.interval)
        self.dictData = self.convertDataFrameToDict(self.dfData)

    def loadData(self):
        try:
            self.updateDictData()
            fileDict = dict()
            with open(self.filePath, 'r') as file:
                fileDict = eval(file.read())

            for key in list(fileDict.keys()):
                if not (key in list(self.dictData.keys())):
                    self.dictData[key] = fileDict[key]

            self.strDict = self.convertDictDataToString(self.dictData)

        except FileNotFoundError:
            self.updateDictData()
            self.strDict = self.convertDictDataToString(self.dictData)

    def updateFile(self):
        self.loadData()
        with open(self.filePath, 'w+') as file:
            file.write(self.strDict)

    def getDict(self):
        self.loadData()
        return self.dictData
