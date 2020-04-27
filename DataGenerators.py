import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone
import DataCollectors as dataCollectors
import dataPreparation


def timedelta_to_minutes(delta):
    deltaMinutes = delta.astype('timedelta64[m]')
    return deltaMinutes.astype('int')


def mulEachElement(arr):
    rv = 1

    for v in arr:
        rv *= v

    return rv


class BackprobegationDataGenerator:
    def __init__(self, stockName, dataCollectorData=True):
        self.stockName = stockName
        self.dataCollector = False
        if dataCollectorData:
            self.dataCollector = dataCollectors.BackPropegationDataCollector(
                self.stockName, f'./Data/{self.stockName}-BackPropegation.txt')
            self.dfData = self.dataCollector.dfData
        else:
            self.dfData = yf.download(
                tickers=stockName, period='60d', interval='2m')
        
        self.dictData = dataPreparation.createDictData(self.dfData, self.dataCollector)

        self.sampleDtype = tf.double
        self.lableDtype = tf.double
        self.sampleShape, self.lableShape = self.calcShapes()

    def plotData(self):
        plt.plot(tf.Variable(self.dictData.values()).numpy())
        plt.show()

    def calcShapes(self):
        data = next(self.__call__())
        return data[0].shape, data[1].shape


    def __call__(self):
        while True:
            for i in range(int(len(self.dictData.keys()) / 2), len(self.dictData.keys())):
                cSampleKeys = self.dictData.keys()[i - int(len(self.dictData.keys()) / 2):i]
                cDate = self.dictData.keys()[i]

                cSamples = []
                for date in cSampleKeys:
                    cSamples.append(self.dictData[date])

                cSamples = tf.reshape(tf.Variable(
                    cSamples), (mulEachElement(tf.Variable(cSamples).shape),))

                for d in self.dictData.keys()[i+1:]:

                    sample = []
                    sample.append(tf.Variable(timedelta_to_minutes(d - cDate)))
                    for j in range(cSamples.shape[0]):
                        sample.append(cSamples[j])

                    sample = tf.reshape(tf.Variable(sample, dtype=self.sampleDtype), (1, mulEachElement(
                        tf.Variable(sample, dtype=self.sampleDtype).shape)))
                    lable = tf.reshape(
                        self.dictData[d], (1, mulEachElement(self.dictData[d].shape)))
             
                    yield sample, lable
