from numpy import *
import os

matrix = mat

def vector(*args):
    return matrix(args).T

def colCount(M): return shape(M)[1]
def rowCount(M): return shape(M)[0]

def vectorLength(v):
    assertIsVector(v)
    return rowCount(v)

def assertIsVector(v):
    assert colCount(v) == 1, 'not a vector, as column count is not exactly one'

def vectorItem(v, index):
    assertIsVector(v)
    assert not (index > rowCount(v)-1), 'index cannot be greater than vector length'
    return v.item(([index][0]))

def setVectorItem(v, index, value):
    assertIsVector(v)
    v[index, 0] = value

def texMatrix(m, isRowsToBeSeparatedWithNewLine = True, decimalCount = 99, roundDecimals = -1):
    result = ""
    for row in range(0, rowCount(m)):
        for col in range(0, colCount(m)):
            value = round(m.item(row, col), decimalCount)
            result += str(value if roundDecimals < 0 else  round(value, roundDecimals))
            if (col < colCount(m) -1):
                result += ' & '
        result += ' \\\\'
        if (isRowsToBeSeparatedWithNewLine == True):
            result += '\n'
    return result

def trimToColumn(M, colIndex):
    entries = []
    for row in range(0, rowCount(M)):
        entry = M.item(row, colIndex)
        entries.append(entry)
    return vector(entries)

def trimVectorToValue(v, value):
    trimmed = []
    for index in range(0, vectorLength(v)):
        entry = vectorItem(v, index)
        if (entry == value):
            trimmed.append(entry)
    return vector(trimmed)

def columns(M):
    vectors = []
    for colIndex in range(0, colCount(M)):
        vectors.append(trimToColumn(M, colIndex))
    return vectors

def clipboard(str):
    os.system("echo '%s' | pbcopy" % str.replace('\\', '\\\\'))

def scalar(M):
    assert colCount(M) == 1, 'col count should be one'
    assert rowCount(M) == 1, 'row count should be one'
    return M.item((0,0))

