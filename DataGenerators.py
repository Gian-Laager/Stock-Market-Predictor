import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone
import DataCollectors as dataCollectors


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
            self.dataCollector = dataCollectors.BackPropegationDataCollector(self.stockName, f'./Data/{self.stockName}-BackPropegation.txt')
        self.dfData = yf.download(
            tickers=stockName, period='60d', interval='2m')
        self.dates = self.extractDates()
        self.dictData = self.ceateDictData()
        self.removeNanFromData()

        self.sampleDtype = tf.double
        self.lableDtype = tf.double
        self.sampleShape = None
        self.lableShape = None
        self.calcShapes()

    def removeNanFromData(self):
        keys = list(self.dictData.keys())

        for key in keys:
            if True in tf.math.is_nan(self.dictData[key]).numpy().flatten().tolist():
                del self.dictData[key]

    def extractDates(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def ceateDictData(self):
        rawData = tf.Variable(self.dfData.to_numpy())

        dictionary = {}
        for i in range(len(self.dates)):
            dictionary[self.dates[i]] = rawData[i]

        if self.dataCollector != False:
            collectorDict = self.dataCollector.getDict()
            for key in list(collectorDict.keys()):
                dictionary[key] = collectorDict[key]

        return dictionary

    def calcShapes(self):
        # this will set the shapes
        next(self.__call__())

    def __call__(self):
        while True:
            for i in range(int(len(list(self.dictData
                                        .keys())) / 2), len(list(self.dictData
                                                                .keys()))):
                cSampleKeys = list(self.dictData
                                .keys())[i - int(len(list(self.dictData
                                                            .keys())) / 2):i]
                cDate = list(self.dictData
                            .keys())[i]

                cSamples = []
                for date in cSampleKeys:
                    cSamples.append(self.dictData
                                    [date])

                cSamples = tf.reshape(tf.Variable(
                    cSamples), (mulEachElement(tf.Variable(cSamples).shape)))

                for d in list(self.dictData
                            .keys())[i+1:]:

                    sample = []
                    sample.append(tf.Variable(timedelta_to_minutes(d - cDate)))
                    for j in range(cSamples.shape[0]):
                        sample.append(cSamples[j])

                    sample = tf.reshape(tf.Variable(sample, dtype=self.sampleDtype), (1, mulEachElement(
                        tf.Variable(sample, dtype=self.sampleDtype).shape)))
                    lable = tf.reshape(
                        self.dictData[d], (1, mulEachElement(self.dictData[d].shape)))
                    self.sampleShape = sample.shape
                    self.lableShape = lable.shape

                    if not (True in tf.math.is_nan(sample).numpy().flatten().tolist()) and not (True in tf.math.is_nan(lable).numpy().flatten().tolist()):
                        yield sample, lable
