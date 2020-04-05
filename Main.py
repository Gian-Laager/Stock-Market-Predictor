import yfinance as yf
import tensorflow as tf
import numpy as np
from pytz import timezone


def arrays_to_dict(keys, data):
    if len(keys) != data.shape[0]:
        raise Exception('keys and data must have the same length')

    dictionary = {}
    for i in range(len(keys)):
        dictionary[keys[i]] = data[i]

    return dictionary


def timedelta_to_minutes(delta):
    deltaMinutes = delta.astype('timedelta64[m]')
    return deltaMinutes.astype('int')


def createDataset(stockName):
    dfData = yf.download(tickers=stockName, period='60d', interval='2m')

    dates = dfData.index.to_numpy()

    for i in range(len(dates)):
        dates[i] = dates[i].to_numpy()

    rawData = tf.Variable(dfData.to_numpy())

    dictData = arrays_to_dict(dates, rawData)

    sampleList = []
    lableList = []

    for i in range(int(len(list(dictData.keys())) / 2), len(list(dictData.keys()))):
        print(i - int(len(list(dictData.keys())) / 2), 'of', len(list(dictData.keys())) - int(len(list(dictData.keys())) / 2))
        cSampleKeys = list(dictData.keys())[i - int(len(list(dictData.keys())) / 2):i]
        cDate = list(dictData.keys())[i]

        cSamples = []
        for date in cSampleKeys:
            cSamples.append(dictData[date])

        for d in list(dictData.keys())[i+1:]:
            sample = []
            sample.append(tf.Variable(timedelta_to_minutes(d - cDate)))
            for i in range(len(cSamples)):
                sample.append(cSamples[i])

            sampleList.append(sample)
            lableList.append(dictData[d])
            

def main():
    stockName = 'AAPL'
    dataset = createDataset(stockName)


if __name__ == "__main__":
    main()
