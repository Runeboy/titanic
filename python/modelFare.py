from project import *

def modelMissingFare(f):
    print 'Modelling missing \'Fare\' values..'
    fareS = f['Fare']

    fareMean = fareS.dropna().mean()

    def fillMissingDeckValue(fare):
        if not isNaN(fare): return fare

        print 'Correcting one fare..'
        return fareMean

    fareS = fareS.apply(fillMissingDeckValue) 

    f['Fare'] =  fareS

if __name__ == "__main__":
    print 'start'

    dataF = loadTestDataFrame()

    modelMissingFare(dataF)

    print 'end'