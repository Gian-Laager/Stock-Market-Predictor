import yfinance as yf
import numpy as np
import tensorflow as tf
import numpy
from threading import Thread
import dataPreparation

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
    def __init__(self, stockName, filePath, period='60d', interval='2m'):
        self.filePath = filePath
        self.stockName = stockName
        self.interval = interval
        self.period = period
        self.dfData = yf.download(
                tickers=self.stockName, period=self.period, interval=self.interval)
        self.dictData = dataPreparation.createDictData(self.dfData)
        self.strDict = None
        self.loadData()
        addThread(self.updateFile())



    def convertDictDataToString(self, dictData):
        strDict = dataPreparation.DateDict()
        for key in dictData.keys():
            strDict[key] = TensroDecoder(dictData[key])

        return str(strDict)

    def loadData(self):
        try:
            fileDict = dataPreparation.DateDict()
            with open(self.filePath, 'r') as file:
                fileDict = eval(file.read())

            self.dictData.update(fileDict)

            self.strDict = self.convertDictDataToString(self.dictData)

        except FileNotFoundError:
            self.strDict = self.convertDictDataToString(self.dictData)

    def updateFile(self):
        self.loadData()
        with open(self.filePath, 'w+') as file:
            file.write(self.strDict)

    def getDict(self):
        self.loadData()
        return self.dictData
