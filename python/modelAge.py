from project import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

survivedAgeSplit = 11
deadAgeSplit = 14

survivalColName = 'IsFemale'

def modelMissingAge(df):
    random.seed(815789) 

    print('Modelling missing \'Age\' values..')

    def modelMissingAgeValue(row):
        if not isNaN(row['Age']): return row['Age']

        survived = row[survivalColName]

        survivalFilteredF =  df[df[survivalColName] == survived]

        adolescentThreshold = survivedAgeSplit if survived else deadAgeSplit
        adolescentFilteredF = survivalFilteredF[survivalFilteredF['Age'] < adolescentThreshold]

        adolescentFraction = float(len(adolescentFilteredF)) / len(survivalFilteredF)

        sampleFromAdolescent = (random.randint(0,100) / 100.0) <= adolescentFraction

        sampleAge =  sampleOnePositiveGauss(adolescentFilteredF['Age']) if sampleFromAdolescent else sampleOnePositiveGauss(survivalFilteredF['Age'])

        return sampleAge

    df['Age'] =  df.apply(modelMissingAgeValue, axis=1)
    random.seed()

def printSurvivorCount():
    survivorCount = len(survivedF.dropna())
    deadCount = len(deadF.dropna())
    print('\nSurvivors: {0}, dead: {1}, ratio: {2}\n'.format(survivorCount, deadCount, (float(survivorCount)/deadCount)))

def processSplit(msg, filename, split, colorBins, colorGauss):

    split = split.dropna()
    binSize = int(split.max() - split.min())

    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_ticks([]) 
    frame1.axes.get_xaxis().tick_bottom()

    frame1.axes.spines['right'].set_visible(False)
    frame1.axes.spines['top'].set_visible(False)
    frame1.axes.spines['left'].set_visible(False)
    plt.xlabel('Age', fontsize=18)
    sigma = split.std()
    mu = split.mean()
    print('\n{0}::\n\tmean:{1}\n\tstd: {2}'.format(msg, mu, sigma))
    count, bins, ignored = plt.hist(split.values, binSize, normed=True, color=colorBins)

    x = np.linspace(split.min(),split.max(),100)
    plt.plot(x,mlab.normpdf(x,split.mean(),split.std()), linewidth=2, color=colorGauss)

    plt.savefig(getImageDataFilepath(filename), bbox_inches='tight')
    plt.clf()[]

if __name__ == "__main__":
    print('start')

    trainDf = loadExpandedTrainDataFrame()

    survivedF = trainDf[trainDf[survivalColName] == 1]
    deadF = trainDf[trainDf[survivalColName] == 0]

    modelMissingAge(trainDf)

    saveDataFrame(trainDf[['Age']], 'modelAge.csv')

    printSurvivorCount()

    print('Making plots..')

    deadAgeS = deadF['Age'].dropna()
    deadAgeUnderThresholdS = deadF[deadF['Age'] < deadAgeSplit ]['Age'].dropna()
    print('\n** Dead under {0} ratio: {1}/{2} = {3}'.format(deadAgeSplit, len(deadAgeUnderThresholdS), len(deadAgeS), float(len(deadAgeUnderThresholdS))/ len(deadAgeS)))
    processSplit(
        'dead',
        'modelAgeDead',
        deadAgeS,
        'pink',
        'red'
        )
    processSplit(
        'dead, under  ' + str(deadAgeSplit ),
        'modelAgeDeadUnder' + str(deadAgeSplit ),

        deadAgeUnderThresholdS,
        'pink',
        'red'
        )

    survivedAgeS = survivedF['Age'].dropna()
    survivedAgeUnderThresholdS = survivedF[survivedF['Age'] < survivedAgeSplit]['Age'].dropna()
    print('\n** Survivors under {0} ratio: {1}/{2} = {3}'.format(survivedAgeSplit, len(survivedAgeUnderThresholdS), len(survivedAgeS), float(len(survivedAgeUnderThresholdS))/ len(survivedAgeS)))
    processSplit(
        'survived',
        'modelAgeSurvived',

        survivedAgeS,
        'skyblue',
        'blue'
        )
    processSplit(
        'survived, under ' + str(survivedAgeSplit),
        'modelAgeSurvivedUnder' + str(survivedAgeSplit),

        survivedAgeUnderThresholdS,
        'skyblue',
        'blue'
        )

    print('end')