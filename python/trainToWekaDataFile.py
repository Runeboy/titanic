from shared import *
from project import *
import numpy as np
import re 
import csv as csv
import pandas as pd

print 'start'

if __name__ == "__main__":
    f = loadExpandedTrainDataFrame()
    dataToArff(
        f,
        'titanic',
        titanicColumnsFilter,

        arffFilepath,
        titanicWekaKeysTypes,

        valueConverters=titanicValueConverters
        )

print 'end'

