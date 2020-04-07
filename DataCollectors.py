import yfinance as yf
import numpy as np
import tensorflow as tf
import numpy


class TensorDecoder:
    def __init__(self, tensor):
        self.tensor = tensor

    def decodeValue(self, strValue):
        listValue = list(strValue)
        for i in range(len(strValue)):
            if listValue[i] == ' ':
                listValue[i] = ','

        rstr = ''
        for char in listValue:
            rstr += char

        return rstr

    def decodeDtype(self, strDtype):
        strDtype = strDtype.split("'")[1]
        strDtype = 'tf.' + strDtype

        return strDtype

    def __repr__(self):
        tValue = str(self.tensor.numpy().tolist())
        dtype = self.decodeDtype(str(self.tensor.dtype))
        shape = str(self.tensor.shape)

        return f'tf.Variable({tValue}, dtype={dtype}, shape={shape})'

class BackpropegationDataCollecotr:
    def __init__(self, stockName, path, prd='60d', intrvl='2m'):
        self.filePath = path
        self.period = prd
        self.interval = intrvl
        self.stockName = stockName
        self.dfData = yf.download(
            tickers=self.stockName, period=self.period, interval=self.interval)
        self.dictData = dict()
        self.loadData()
        self.updateFile()

    def extractDatesFromDf(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def updateDataFrameData(self):
        self.dfData = yf.download(
            tickers=self.stockName, period=self.period, interval=self.interval)

    def loadData(self):
        try:
            with open(self.filePath, 'r') as file:
                strDict = file.read()
                self.dictData = eval(strDict)

            self.updateDataFrameData()
            newDictData = self.convertDataFrameToDict()
            newDictData = self.removeNanFromData(newDictData)
            for newKey in list(newDictData.keys()):
                newDictData[newKey] = TensorDecoder(newDictData[newKey])
                self.dictData[newKey] = newDictData[newKey]

        except FileNotFoundError:
            self.updateDataFrameData()
            self.dictData = self.convertDataFrameToDict()
            self.dictData = self.removeNanFromData(self.dictData)
            for key in list(self.dictData.keys()):
                self.dictData[key] = TensorDecoder(self.dictData[key])

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
            file.write(str(self.dictData))
