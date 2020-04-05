import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone


def timedelta_to_minutes(delta):
    deltaMinutes = delta.astype('timedelta64[m]')
    return deltaMinutes.astype('int')


def mulEachElement(arr):
    rv = 1

    for v in arr:
        rv *= v

    return rv


class DataGenarator:
    def __init__(self, stockName):
        self.stockName = stockName
        self.dfData = yf.download(
            tickers=stockName, period='60d', interval='2m')
        self.dates = self.extractDates()
        self.dictData = self.convertDataFrameToDict()
        self.sampleShape, self.lableShape = self.calcShapes()
        self.sampleDtype = tf.double
        self.lableDtype = tf.double

    def extractDates(self):
        dates = self.dfData.index.to_numpy()

        for i in range(len(dates)):
            dates[i] = dates[i].to_numpy()

        return dates

    def convertDataFrameToDict(self):
        rawData = tf.Variable(self.dfData.to_numpy())

        dictionary = {}
        for i in range(len(self.dates)):
            dictionary[self.dates[i]] = rawData[i]

        return dictionary

    def calcShapes(self):
        sKeys = list(self.dictData.keys())[
            :int(len(list(self.dictData.keys())) / 2)]
        s = []
        for date in sKeys:
            s.append(self.dictData
                     [date])
        s = np.array(s)

        sampleShape = tf.TensorShape(mulEachElement(s.shape) + 1)
        lableShape = self.dictData[list(self.dictData.keys())[0]].shape

        return sampleShape, lableShape

    def __call__(self):
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

                yield tf.Variable(sample, dtype=self.sampleDtype), self.dictData[d].astype(self.lableDtype)


def main():
    STOCKNAME = 'AAPL'

    dataGenarator = DataGenarator(STOCKNAME)

    dataset = tf.data.Dataset.from_generator(
        dataGenarator, (dataGenarator.sampleDtype, dataGenarator.lableDtype), (dataGenarator.sampleShape, dataGenarator.lableShape))


if __name__ == "__main__":
    main()
