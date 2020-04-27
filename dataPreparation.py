import numpy as np
import tensorflow as tf

class DateDict(dict):
    def __init__(self, iterable={}):
        super().__init__(iterable)

    def __getitem__(self, key):
        if type(key) != slice:
            return super().__getitem__(key)

        else:
            keys = np.sort(list(super().keys()))[list(super().keys()).index(key.start): list(super().keys()).index(key.stop)]
            retDict = DateDict()
            for k in keys:
                retDict[k] = super().__getitem__(k)
            
            return retDict

    def tolist(self):
        retList = []
        for k in list(super().keys()):
            retList.append(super().__getitem__(k))
        return retList

    def keys(self):
        return np.sort(list(super().keys()))

    def values(self):
        return list(super().values())


def extractDates(dataFrame):
    dates = dataFrame.index.to_numpy()

    for i in range(len(dates)):
        dates[i] = dates[i].to_numpy()

    return dates

def createDictData(dfData, dataCollector=False, dictionary=DateDict()):
    rawData = dfData.to_numpy()

    dates = extractDates(dfData)

    for i in range(len(dates)):
        dictionary[dates[i]] = tf.Variable(np.average(rawData[i][:len(rawData[i]) -2]))

    if dataCollector != False:
        collectorDict = dataCollector.getDict()
        for key in list(collectorDict.keys()):
            dictionary[key] = collectorDict[key]

    return removeNanFromData(dictionary)

def removeNanFromData(dictData):
        keys = list(dictData.keys())

        for key in keys:
            if True in tf.math.is_nan(dictData[key]).numpy().flatten().tolist():
                del dictData[key]
        return dictData