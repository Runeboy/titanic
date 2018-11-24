from shared import *
from project import *
import numpy as np
import re 
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def modelMissingDeck(f):
    random.seed(2919285391)

    print 'Modelling missing \'DeckNumber\' values..'
    deckS = f['Deck']
    nullCountBefore = deckS.isnull().sum()
    deckS = deckLettersToNumbers(deckS)
    nullCountAfter = deckS.isnull().sum()
    assert nullCountBefore == nullCountAfter

    mean = deckS.dropna().mean()
    std  = deckS.dropna().std()

    def fillMissingDeckValue(deckNumber):

        return deckNumber if not isNaN(deckNumber) else sampleGauss(deckS.dropna(), min=deckS.min(), max=deckS.max())
    deckS = deckS.apply(fillMissingDeckValue)

    f['DeckNumber'] = deckS

    random.seed()

if __name__ == "__main__":
    print 'start'

    dataF = loadExpandedTrainDataFrame()

    modelMissingDeck(dataF)

    print dataF[['Deck', 'DeckNumber']]

    plotGauss(
        dataF['DeckNumber'].dropna(),
        xTitle='Deck ',
        filename='modelDeck'

        )

    saveDataFrame(dataF[['Deck']], 'modelDeck.csv')

    print 'end'