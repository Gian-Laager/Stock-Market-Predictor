import yfinance as yf
import numpy as np
import tensorflow as tf
import numpy
from threading import Thread


def addThread(*funcs):
    for func in funcs:
        thread = Thread(target=func)
        thread.start()


class TensorDecoder:
    def __init__(self, tensor):
        self.tensor = tensor

    def decodeDtype(self, strDtype):
        strDtype = strDtype.split("'")[1]
        strDtype = 'tf.' + strDtype

        return strDtype

    def __repr__(self):
        tValue = str(self.tensor.numpy().tolist())
        dtype = self.decodeDtype(str(self.tensor.dtype))
        shape = str(self.tensor.shape)

        return f"'tf.Variable({tValue}, dtype={dtype}, shape={shape})'"


class BackpropegationDataCollecotr:
    def __init__(self, stockName, path, prd='60d', intrvl='2m'):
        self.filePath = path
        self.period = prd
        self.interval = intrvl
        self.stockName = stockName
        self.dfData = yf.download(
            tickers=self.stockName, period=self.period, interval=self.interval)
        self.__dictData = dict()
        self.loadData()
        self.updateFile()

        addThread(self.threadLoopFunction)

    def threadLoopFunction(self):
        while True:
            self.updateFile

    def extractDatesFromDf(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def updateDataFrameData(self):
        self.dfData = yf.download(
            tickers=self.stockName, period=self.period, interval=self.interval)

    def formatStrDict(self, strDict):
        strDict = strDict.replace(' ', '')
        strDict = strDict.replace('\n', '')        
        return strDict

    def evalStrData(self, strDict):
        dictData = eval(strDict)
        for key in list(dictData.keys()):
            try:
                dictData[key] = eval(dictData[key])
            except SyntaxError:
                del dictData[key]

        return dictData

    def loadData(self):
        try:
            with open(self.filePath, 'r') as file:
                strDict = file.read()
                strDict = self.formatStrDict(strDict)
                if len(strDict) > 0:
                    self.__dictData = self.evalStrData(strDict)
                else:
                    self.__dictData = dict()
            
            self.updateDataFrameData()
            newDictData = self.convertDataFrameToDict()
            newDictData = self.removeNanFromData(newDictData)
            for newKey in list(newDictData.keys()):
                newDictData[newKey] = TensorDecoder(newDictData[newKey])
                self.__dictData[newKey] = newDictData[newKey]


        except FileNotFoundError:
            self.updateDataFrameData()
            self.__dictData = self.convertDataFrameToDict()
            self.__dictData = self.removeNanFromData(self.__dictData)
            for key in list(self.__dictData.keys()):
                self.__dictData[key] = TensorDecoder(self.__dictData[key])

    def convertDataFrameToDict(self):
        rawData = tf.Variable(self.dfData.to_numpy())
        dates = self.extractDatesFromDf()

        dictionary = {}
        for i in range(len(dates)):
            dictionary[dates[i]] = rawData[i]

        return dictionary

    def removeNanFromData(self, dictData):
        keys = list(dictData.keys())

        for key in keys:
            if True in tf.math.is_nan(dictData[key]).numpy().flatten().tolist():
                del dictData[key]

        return dictData

    def updateFile(self):
        self.loadData()
        with open(self.filePath, 'w+') as file:
            file.write(str(self.__dictData))

    def getDict(self):
        self.updateFile()
        debug = str(self.__dictData)
        return self.__dictData
