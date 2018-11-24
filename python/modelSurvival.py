from project import *
import numpy as np

def modelMissingSurvival(df):
    assert 'Survived' not in df, 'Survived column already exists!'

    df['Survived'] = np.nan

    print 'Modelling missing survival values..'

    def modelMissingSurvivalValue(row):
        if not isNaN(row['Survived']): return row['Survived']
        return 1 if row['Sex'] == 'female' else 0

    df['Survived'] =  df.apply(modelMissingSurvivalValue, axis=1)

if __name__ == "__main__":
    print 'start'

    testF = loadTestDataFrame()

    print len(testF)

    modelMissingSurvival(testF)
    saveDataFrame(testF[['Survived']], 'testModelSurvived.csv')

    print 'end'

