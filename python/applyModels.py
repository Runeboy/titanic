from project import *

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from modelAge import modelMissingAge
from modelDeck import modelMissingDeck
from modelDeckCabin import modelMissingDeckCabin
from modelFare import modelMissingFare

def ensureNoNullCols(f, colName):
    nullCount = seriesNullCount(f[colName])
    assert nullCount == 0, 'some missing \'{0}\' cols have not been filled in (count:{1})'.format(colName, nullCount)

def applyModels(f):
    modelMissingAge(f)
    ensureNoNullCols(f, 'Age')

    modelMissingDeck(f)
    ensureNoNullCols(f, 'DeckNumber')

    modelMissingDeckCabin(f)
    ensureNoNullCols(f, 'DeckCabin')

    modelMissingFare(f)
    ensureNoNullCols(f, 'Fare')

if __name__ == "__main__":
    print 'start'

    trainF = loadExpandedTrainDataFrame()
    testF  = loadExpandedTestDataFrame()

    print '\n** Applying missing attribute fill-in models to train..'
    applyModels(trainF)

    print '\n** Applying missing attribute fill-in models to test..'
    applyModels(testF)

    colsToKeep = ['Survived', 'Age', 'Pclass', 'NameLength', 'Fare', 'DeckCabin', 'DeckNumber', 'IsFemale'] 
    trimmedTrainF = trainF[colsToKeep]
    trimmedTestF  = testF[colsToKeep]

    assert anySeriesNullCount(trimmedTrainF) == 0, 'Train data: count of rows with one or more null-valued columns did not equal zero '
    assert anySeriesNullCount(trimmedTrainF) == 0, 'Test data:count of rows with one or more null-valued columns did not equal zero '

    saveDataFrame(trimmedTrainF, 'trainTrimmedNoneMissing.csv', sep=',')
    saveAsArff(trimmedTrainF, "trainTrimmedNoneMissing", "trainTrimmedNoneMissing.arff")

    saveDataFrame(trimmedTestF, 'testTrimmedNoneMissing.csv', sep=',')
    saveAsArff(trimmedTestF, "testTrimmedNoneMissing", "testTrimmedNoneMissing.arff")

    yName = 'Survived'

    def getAccuracy(truth, predictedTruth):
            assert len(predictedTruth) == len(truth)

            correctPredictionCount =  (truth == predictedTruth).map({True: 1, False:0}).sum()
            accuracy = float(correctPredictionCount) / len(testF)

            return accuracy

    def predictTruth(trainF, testF, truthColName, forestParam):

            forest = RandomForestClassifier(n_estimators=forestParam)
            trainTruth = trainF[truthColName].values
            forest = forest.fit(
                dropColumn(trainF, truthColName).values,
                trainTruth
                )

            testTruth = testF[truthColName]

            predictedTruth = forest.predict(
                dropColumn(testF, truthColName).values
                )

            assert len(predictedTruth) == len(testTruth )

            correctPredictionCount =  (testTruth == predictedTruth).map({True: 1, False:0}).sum()
            accuracy = float(correctPredictionCount) / len(testF)

            return accuracy

    foldCount = 10
    def runCrossValidation(f, forestEstimatorCountParam):
        setSize = len(f) / foldCount 
        print "Splitting, total:{0} -> {1} sets, each size {2} (last takes the leftovers) ".format(len(f), foldCount, setSize)
        fSets = []
        for foldIndex in range(0, foldCount):
            startIndex = foldIndex * setSize
            endIndex = len(f) if (foldIndex == foldCount-1) else startIndex + setSize

            fSets.append(f[startIndex : endIndex])

        map = 0.0

        for testIndex, testSet  in enumerate(fSets):

            trainSets = fSets[0:testIndex] + fSets[testIndex+1:foldCount-1]
            trainSet = pd.concat(trainSets)

            truth = testSet[yName].values 
            accuracy = predictTruth(trainSet, testSet, yName, forestEstimatorCountParam)

            map += accuracy / foldCount

        print 'MAP Accuracy for all folds: {0}'.format(map)

        return map

    sweepRange = range(1,300)

    sweepResults = pd.DataFrame(sweepRange, index=sweepRange, columns=['forestEstimatorParam']).set_index('forestEstimatorParam')
    sweepResults['map'] = np.nan

    random.seed(2182858637)
    for sweepValue, row in sweepResults.iterrows():        
        print '--Running cross validation with sweep value: ' + str(sweepValue)
        map = runCrossValidation(trimmedTrainF, sweepValue)

        row['map'] = map 

    print sweepResults
    random.seed()

    saveDataFrame(sweepResults, 'predictionParamSweep.csv', sep=',')

    print '\n---------Best MAP for param sweep:--------'

    maxMapSweepParam = sweepResults['map'].idxmax()
    maxMap = sweepResults.ix[maxMapSweepParam ]['map']

    print 'Max MAP: {0}, for row with index (forest param): {1}'.format(maxMap, maxMapSweepParam)

    print '------------------\n'
    print '\nTraining...'
    forest = RandomForestClassifier(n_estimators=maxMapSweepParam)

    forest = forest.fit(
        dropColumn(trimmedTrainF, yName).values,
        trimmedTrainF[yName].values
        )

    print 'Predicting...'
    output = forest.predict(
        dropColumn(trimmedTestF, yName).values
        )

    assert len(output) == len(trimmedTestF)

    trimmedTestF.is_copy = False
    trimmedTestF['Survived'] = pd.Series(output, index=trimmedTestF.index)
    saveDataFrame(trimmedTestF, 'testPredicted.csv', sep=',')
    saveDataFrame(trimmedTestF[['Survived']], 'testPredictedSurvival.csv', sep=',')

    print 'end'

