from project import *

survivalColName = 'IsFemale'

def deck(f, deck):
    return f[f['Deck'] == deck]

def modelMissingDeckCabin(f, binSize=10):
    random.seed(491828481)

    print 'Modelling missing \'DeckCabin\' values..'

    assumedSurvivorMean = 27
    assumedDeadMean = 30

    survivorDeckCabinS = survivors(f, survivalColumnName=survivalColName)['DeckCabin']
    survivorBellRightHalfS = survivorDeckCabinS [survivorDeckCabinS >= assumedSurvivorMean].dropna()
    survivorSymmetricAtMeanS = survivorBellRightHalfS.append(2 * assumedSurvivorMean - survivorBellRightHalfS)
    plotGauss(survivorDeckCabinS, filename='modelDeckCabinSurvivors', binSize=binSize, gaussMean=survivorSymmetricAtMeanS.mean(), gaussStd=survivorSymmetricAtMeanS.std(), gaussScale=1.3, colorGauss='blue', colorBins='skyblue')

    deadDeckCabinS = nonsurvivors(f, survivalColumnName=survivalColName)['DeckCabin']
    deadBellRightHalfS = deadDeckCabinS [deadDeckCabinS >= assumedDeadMean].dropna()
    deadSymmetricAtMeanS = deadBellRightHalfS.append(2 * assumedDeadMean - deadBellRightHalfS)
    plotGauss(deadDeckCabinS , filename='modelDeckCabinDead', binSize=binSize, gaussMean=deadSymmetricAtMeanS.mean(), gaussStd=deadSymmetricAtMeanS.std(), gaussScale=1.3, colorGauss='red', colorBins='pink')

    def modelMissingDeckCabinValue(row):
        if not isNaN(row['DeckCabin']): return row['DeckCabin']

        survived = row[survivalColName]
        return sampleGauss(
            survivorSymmetricAtMeanS if survived else deadSymmetricAtMeanS,
            min=0
            )

    f['DeckCabin'] = f.apply(modelMissingDeckCabinValue, axis=1)
    random.seed()

if __name__ == "__main__":
    print 'start'

    f = loadExpandedTrainDataFrame()

    cabinS =  f['DeckCabin']

    binSize = 10

    print 'Makin cabin plots for each deck..'
    decks = ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    for d in decks:
        plotGauss(
            survivors(deck(f, d), survivalColumnName=survivalColName)['DeckCabin'].dropna(),
            xTitle='Deck cabin',
            binSize=binSize,
            colorGauss='blue', colorBins='skyblue',
            filename='modelDeck{0}CabinSurvivors'.format(d)
            )
        plotGauss(
            nonsurvivors(deck(f, d), survivalColumnName=survivalColName)['DeckCabin'].dropna(),
            xTitle='Deck cabin',
            binSize=binSize,
            colorGauss='red', colorBins='pink',
            filename='modelDeck{0}CabinDead'.format(d)
            )

    print 'Makin cabin plots for all decks..'
    plotGauss(
        survivors(f, survivalColumnName=survivalColName)['DeckCabin'].dropna(),
        xTitle='Deck cabin',
        binSize=binSize,
        colorGauss='blue', colorBins='skyblue',
        filename='modelDeckCabinSurvivorsPreCorrect'
        )
    plotGauss(
        nonsurvivors(f, survivalColumnName=survivalColName)['DeckCabin'].dropna(),
        xTitle='Deck cabin',
        binSize=binSize,
        colorGauss='red', colorBins='pink',
        filename='modelDeckCabinDeadPreCorrect'
        )

    print 'Making model..'
    modelMissingDeckCabin(f, binSize=binSize)

    saveDataFrame(f[['DeckCabin']], 'modelDeckCabin.csv')

    print 'end'