from shared import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import pandas as pd
import re 
import matplotlib.mlab as mlab

def plotGauss(series, binSize=None, colorGauss='green', colorBins='lightgreen', xTitle='', filename=None, clearAfter=True, hideBins=False, gaussMean=None, gaussStd=None, gaussScale=1):
    series = series.dropna()

    if binSize is None: binSize = int(series.max() - series.min())

    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_ticks([]) 
    frame1.axes.get_xaxis().tick_bottom()

    frame1.axes.spines['right'].set_visible(False)
    frame1.axes.spines['top'].set_visible(False)
    frame1.axes.spines['left'].set_visible(False)

    plt.xlabel(xTitle, fontsize=16)

    if not hideBins:
        count, bins, ignored = plt.hist(series.values, binSize, normed=True, color=colorBins)

    x = np.linspace(series.min(),series.max(),100)
    plt.plot(x, gaussScale * mlab.normpdf(x,
                             series.mean() if gaussMean is None else gaussMean,
                             series.std() if gaussStd is None else gaussStd)
             , linewidth=2, color=colorGauss)

    if filename is not None:
        plt.savefig(getImageDataFilepath(filename), bbox_inches='tight')
    else:
        plt.show()

    if clearAfter: plt.clf()

def isDict(value):
    return isinstance(value, dict)

def isList(value):
    return isinstance(value, list)

def isString(value):
    return isinstance(value, basestring)

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

WekaType = enum(numeric='numeric', string='string')

def intTryParse(value):
    try:
        return int(value)
    except ValueError:
        return nan

def wekaValue(value, valueWekaDataType):
    isMappedDataType = isDict(valueWekaDataType)

    if isMappedDataType:

        key = str(value)
        if not key in valueWekaDataType:
            raise Exception('cannot map value \'{0}\' ({1}) to weka data using {2}'.format(key, type(key), valueWekaDataType));
        value = valueWekaDataType[key]
    if valueWekaDataType is WekaType.string:
        value = value.replace('"', '\\"')
        value = '"' + value + '"'
    return str(value)

def isNaN(num):
    return num != num

def sampleOnePositiveGauss(ageSeries):
    mu = ageSeries.mean()
    sigma  = ageSeries.std()
    value = None
    while value is None or value < 0:
        value = np.random.normal(mu, sigma, 1)[0]
    return value

def sampleGauss(ageSeries, min=None, max=None):
    mu = ageSeries.mean()
    sigma  = ageSeries.std()
    value = None
    while value is None or ((min is not None and value < min) or (max is not None and value > max)):
        value = np.random.normal(mu, sigma, 1)[0]
    return value

def transformAttributes(dataFrame, columnsFilter, keyTypes, valueConverters={}):
    print('Expanding data frame with derived columns..')
    for columnKeyOrFilter in columnsFilter:
        isMappedKey = isDict(columnKeyOrFilter)
        key = None
        if (isMappedKey):
            mapFromKey = list(columnKeyOrFilter.keys())[0] if isMappedKey else columnKeyOrFilter
            mapToKey =  columnKeyOrFilter[mapFromKey] if isMappedKey else columnKeyOrFilter

            dataFrame[mapToKey] = dataFrame[mapFromKey]
            key = mapToKey
        else:
            key = columnKeyOrFilter

        if key in valueConverters:

            converter = valueConverters[key]
            dataFrame[key] = converter(dataFrame[key])

def writeWekaAttribute(file, key, keyTypes):
    dataType = keyTypes[key]
    if isDict(dataType):
        dataType = dataType.values()
    dataType = wekaDataType(dataType)
    file.write('@attribute \'{0}\' {1}\n'.format(key, dataType))

def wekaDataType(list):
    return str(list).replace('[', '{').replace(']', '}').replace('\'', '')

def makeScatterAndSaveToFile(headers, datas, filepath):

    df = pd.DataFrame(datas, columns=['A','B'])
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filepath)

def csvToLists(filepath):
    datas = []
    with open(filepath, 'rb') as csvFile:
        reader = csv.reader(open(filepath, 'rb'), skipinitialspace=True)
        for index, data in enumerate(reader):
            datas.append(data)
    return datas

def modelAge(ageCols):
    return ageCols

def genderToBinary(genderCols):
    return genderCols.apply(lambda gender: 1 if gender == 'female' else 0 if gender == 'male' else Exception('Could not convert gender value \''+str(gender)+'\'.'))

def namesToLength(nameCols):
    return nameCols.apply(len)

def cabinToDeckNominal(cabinStrCols):
    def colApply(cabinStr):
        if cabinStr is NaN: return NaN
        validDeckCabinPairs = re.sub('[A-Z][^\d]', '', cabinStr).split()
        if (len(validDeckCabinPairs) > 0):

            return re.sub(r'[^A-Z]', '', cabinStr)[0]

        return NaN
    return cabinStrCols.apply(colApply)

def trimToNumbers(cabinStrCols):
    def colApply(cabinStr):
        if cabinStr is NaN: return NaN
        validDeckCabinPairs = re.sub('[A-Z][^\d]', '', cabinStr).split()
        if (len(validDeckCabinPairs) > 0):

            cabinNumberStr = re.sub('[^\d]', '', validDeckCabinPairs[0]) 
            return intTryParse(cabinNumberStr)
        return NaN
    return cabinStrCols.apply(colApply)

def getDataFilepath(filename):
    return dataFilepath + filename;

def getImageDataFilepath(filename):
    return imageDataPath +  filename;

def survivors(f, survivalColumnName='Survived'):
    return f[f[survivalColumnName] == 1]

def nonsurvivors(f, survivalColumnName='Survived'):
    return f[f[survivalColumnName] == 0]

def loadDataFrame(filename, sep='\t'):
    return pd.read_csv(getDataFilepath(filename), header=0, sep=sep).set_index('PassengerId')

def loadTrainDataFrame():
    return pd.read_csv(trainFilepath, header=0).set_index('PassengerId')

def loadTestDataFrame():

    f = pd.read_csv(testFilepath, header=0).set_index('PassengerId')

    return f

def loadExpandedTrainDataFrame():
    f = loadTrainDataFrame()
    transformAttributes(f, titanicColumnsFilter, titanicWekaKeysTypes, valueConverters=titanicValueConverters)
    return f

def loadExpandedTestDataFrame():
    f = loadTestDataFrame()
    f['Survived'] = np.nan
    transformAttributes(f, titanicColumnsFilter, titanicWekaKeysTypes, valueConverters=titanicValueConverters)
    return f

def dropColumn(frame, colName):
    return frame.drop(colName, axis=1)

def seriesNullCount(series):
    return series.isnull().sum()

def anySeriesNullCount(frame):
    return frame.isnull().any(axis=1).apply(lambda b: 1 if b else 0).sum()

def saveDataFrame(frame, filename, sep='\t'):
    frame.to_csv(getDataFilepath(filename), sep=sep, encoding='utf-8')

dataFilepath = '../data/';
imageDataPath = dataFilepath + 'image/'
trainFilepath = getDataFilepath('train.csv')
testFilepath = getDataFilepath('test.csv')

scatterFilepath = getDataFilepath('scatter_matrix.png')

titanicIgnoreKeys = ['PassengerId', 'Name', 'Ticket'] 

arffFilepath =  getDataFilepath('train.arff')

titanicBinomialDataType = {'0': 'no', '1': 'yes'}

titanicWekaKeysTypes = {
    'PassengerId': WekaType.numeric,
    'Survived': WekaType.numeric,
    'Pclass': WekaType.numeric,
    'Name': WekaType.string,
        'NameLength': WekaType.numeric,
    'Sex': ['male', 'female'],
    'IsFemale': WekaType.numeric,
    'Age': WekaType.numeric,
    'SibSp': WekaType.numeric,
    'Parch': WekaType.numeric,
    'Ticket': WekaType.string,
    'Fare': WekaType.numeric,
    'Cabin': WekaType.string,
    'Deck': ['A','B','C','D', 'E', 'F', 'G', 'T'],
    'DeckNumber': WekaType.numeric,
    'DeckCabin': WekaType.numeric,
    'Embarked': ['S', 'C', 'Q'] 
}

titanicColumnsFilter = [

    'Pclass',

    {'Name': 'NameLength'},
    'Sex',
    {'Sex': 'IsFemale'},
    'Age',
    'SibSp',
    'Parch',

    'Fare',
    {'Cabin': 'DeckCabin'},
    {'Cabin': 'Deck'},
    {'Deck': 'DeckNumber'}, 
    'Embarked',
    'Survived',
    ]

def deckLettersToNumbers(deckCols):
    return deckCols.map({'T':0, 'A':1, 'B': 2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7})

titanicValueConverters = {
    'Deck': cabinToDeckNominal,
    'NameLength': namesToLength,
    'DeckCabin': trimToNumbers,
    'Age': modelAge,
    'IsFemale': genderToBinary,
    'DeckNumber': deckLettersToNumbers
    }

def dataToArff(dataFrame, relationName, outFilepath, keyTypes = titanicWekaKeysTypes, columnsFilter = titanicColumnsFilter, valueConverters={}):
    with open(outFilepath, 'w') as file:
        colKeysToWrite = []

        for columnKeyOrFilter in columnsFilter:
            isMappedKey = isDict(columnKeyOrFilter)
            key = None
            if (isMappedKey):
                mapFromKey = columnKeyOrFilter.keys()[0] if isMappedKey else columnKeyOrFilter
                mapToKey =  columnKeyOrFilter[mapFromKey] if isMappedKey else columnKeyOrFilter

                dataFrame[mapToKey] = dataFrame[mapFromKey]
                key = mapToKey
            else:
                key = columnKeyOrFilter

            if key in valueConverters:

                converter = valueConverters[key]
                dataFrame[key] = converter(dataFrame[key])

            colKeysToWrite.append(key)

        file.write('@relation {0}\n\n'.format(relationName))
        for key in colKeysToWrite:
            writeWekaAttribute(file, key, keyTypes)

        file.write('\n\n@data\n')
        for index, row in dataFrame.iterrows():        

            line = ''
            for key in colKeysToWrite:
                value = row[key]
                isDataMissing = value is NaN or isNaN(value) or value is '' or (isinstance(value, basestring) and value.isspace())

                if not isDataMissing:
                    dataType = keyTypes[key]
                    isEnumButNotValidMember = isList(dataType) and value not in dataType
                    if isEnumButNotValidMember:

                        raise Exception('Value \'{0}\' not in type {1}.'.format(value, dataType))

                if (len(line) > 0):
                    line += ','
                line += '?' if isDataMissing else wekaValue(value, dataType)
            file.write(line + '\n')

def saveAsArff(dataFrame, relationName, outFilename, targetKey='Survived', keyTypes = titanicWekaKeysTypes):
    outFilepath = getDataFilepath(outFilename)
    with open(outFilepath, 'w') as file:
        file.write('@relation {0}\n\n'.format(relationName))
        for key in dataFrame:
            if key != targetKey: writeWekaAttribute(file, key, keyTypes)
        writeWekaAttribute(file, targetKey, keyTypes)

        def writeKey(key, line):
            value = row[key]
            isDataMissing = value is NaN or isNaN(value) or value is '' or (isinstance(value, basestring) and value.isspace())

            dataType = keyTypes[key]

            if (len(line) > 0):
                line += ','
            line += '?' if isDataMissing else wekaValue(value, dataType)

            return line

        file.write('\n\n@data\n')
        for index, row in dataFrame.iterrows():        

            line = ''
            for key in dataFrame:
                if key != targetKey: line = writeKey(key, line)
            line = writeKey(targetKey, line)
            file.write(line + '\n')

