import pandas as pd
from typing import List, Tuple
from pandas.api.types import is_list_like
import warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.stats import pearsonr

import math


def _get_32_bit_dtype(x):
    dtype = x.dtype
    if dtype.name.startswith("float"):
        redn_dtype = "float32"
    elif dtype.name.startswith("int"):
        redn_dtype = "int32"
    else:
        redn_dtype = None
    return redn_dtype

def add_lags(
    df: pd.DataFrame,
    lags: List[int],
    column: str,
    ts_id: str = None,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe
    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(lags), "`lags` should be a list of all required lags"
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        # warnings.warn(
        #     "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        # )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df[column].shift(l).astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column]
                .shift(l)
                .astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
            }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features
    
    
def _selectTestClimate(protoClimate, testPercent = 0.15, randomSelect = False):
    testClimateCount = math.floor(len(protoClimate) * testPercent)
    testClimate = [0] * (len(protoClimate) - testClimateCount) + [1] * testClimateCount
    if randomSelect == True:
        random.shuffle(testClimate)
    return testClimate

def sequencesGeneration_legacy(data, endIndex, windowSize, startIndex = 0, stride = 1, downSample = 1):
    numSequence = (endIndex + 1) - (windowSize - 1)
    sequencesIndex = (
        startIndex + # start index of the whole data in use
        np.tile(np.arange(windowSize, step = downSample), (numSequence, 1)) + # window
        np.tile(np.arange(numSequence, step = stride), (windowSize, 1)).T # sequence
    )
    return data[sequencesIndex]

def sequencesGeneration(df, lag, featureList, target):
    target = [target]
    sequences = np.empty((len(df) - (lag[-1]), len(lag) + 1, 1))
    for variable in (target + featureList):
        dataFull_lag = add_lags(df, lags = lag, column = variable)[0]
        dataFull_lag = dataFull_lag.dropna()

        dropList = featureList + target
        dropList.remove(variable)
        dataFull_lag = dataFull_lag.drop(columns = dropList, axis = 1)

        dataFull_lag = dataFull_lag[dataFull_lag.columns[::-1]] # data point itself and all lags

        sequences = np.concatenate((sequences, dataFull_lag.values[:, :, np.newaxis]), axis = -1)
    sequences = sequences[:, :, 1:]
    return sequences

def splitData(sequences, valProportion, testProportion, yCount, shuffle = False, ifMakeTest = False):
    if shuffle == True:
        np.random.shuffle(sequences)

    if ifMakeTest == True:
        valCount = int(np.floor(sequences.shape[0] * valProportion))
        testCount = int(np.floor(sequences.shape[0] * testProportion))
        trainCount = sequences.shape[0] - valCount - testCount

        trainX = sequences[:trainCount, :-yCount, 1:]
        trainY = sequences[:trainCount, -yCount:, [0]]
        valX = sequences[trainCount: trainCount + valCount, :-yCount, 1:]
        valY = sequences[trainCount: trainCount + valCount, -yCount:, [0]]
        testX = sequences[trainCount + valCount : trainCount + valCount + testCount, :-yCount, 1:]
        testY = sequences[trainCount + valCount : trainCount + valCount + testCount, -yCount:, [0]]

    elif ifMakeTest == False:
        valCount = int(np.floor(sequences.shape[0] * valProportion))
        trainCount = sequences.shape[0] - valCount

        trainX = sequences[:trainCount, :-yCount, 1:] # the time lag of energy/heat should not be included
        trainY = sequences[:trainCount, -yCount:, [0]]
        valX = sequences[trainCount:, :-yCount, 1:] # the time lag of energy/heat should not be included
        valY = sequences[trainCount:, -yCount:, [0]]

    print('Dims: (sample, sequenceLen, feature)')
    print('trainX shape:', trainX.shape, ' trainY shape:', trainY.shape)
    print('valX shape:', valX.shape, ' valY shape:', valY.shape)

    if ifMakeTest == True:
        print('testX shape:', testX.shape, ' testY shape:', testY.shape)
        return trainX, trainY, valX, valY, testX, testY
    elif ifMakeTest == False:
        return trainX, trainY, valX, valY

def splitData_biRNN(sequences, valProportion, testProportion, yCount, shuffle = False):

    if shuffle == True:
        np.random.shuffle(sequences)

    # for convenience, the length of each sequence should an odd number, the target timestamp is in the middle
    sequenceLen = sequences.shape[1]
    if sequenceLen % 2 == 0:
        raise ValueError("For convenience, please make sure the length of time sequence is an odd number.")
    yLoc = int((sequenceLen + 1) / 2)
    
    # split train and val
    valCount = int(np.floor(sequences.shape[0] * valProportion))
    trainCount = sequences.shape[0] - valCount
    
    # split x and y
    trainX = sequences[:trainCount, :, 1:] # select features with all timestamps
    trainY = sequences[:trainCount, (yLoc - 1) : yLoc, [0]] # select target with the timestamp in the middle
    valX = sequences[trainCount:, :, 1:] 
    valY = sequences[trainCount:, (yLoc - 1) : yLoc, [0]]

    print('Dims: (sample, sequenceLen, feature)')
    print('trainX shape:', trainX.shape, ' trainY shape:', trainY.shape)
    print('valX shape:', valX.shape, ' valY shape:', valY.shape)

    return trainX, trainY, valX, valY
    
def makeDatasets(protoClimate, data, lag, target, weatherFeatureList, splitFunc,
                 vtPercent = 0.15, allInTrain = False, shuffle = False):
    # The data from prototype-weather pair is processed seperately the combined.
    # Test prototype-weather pairs are eliminated before using splitData.

    sequenceList_train = []
    sequenceList_test = []
    climateList_test = []

    # make sequences from df and concat them, if they are for train
    for cli, ifTest in zip(protoClimate, _selectTestClimate(protoClimate)):

        # put all weather into training, only use it for tract level estimation and eval
        if allInTrain == True:
            ifTest = 0

        dataSelect = data[data['Climate'] == cli]
        dataSelectShort = dataSelect[[target] + weatherFeatureList]
        sequences = sequencesGeneration(dataSelectShort, lag, weatherFeatureList, target)

        if ifTest == 0:
            sequenceList_train.append(sequences)
        elif ifTest == 1:
            sequenceList_test.append(sequences)
            climateList_test.append(cli)
    sequences_train = np.concatenate(sequenceList_train, axis = 0)

    # split sequneces into datasets
    trainX, trainY, valX, valY = splitFunc(sequences_train, vtPercent, vtPercent, 1, shuffle = shuffle)
    print('The length of testSequenceList is ', len(sequenceList_test))
    if allInTrain == False:
        print('Each sequence set in testSequenceList is shaped ', sequenceList_test[0].shape)
    return trainX, trainY, valX, valY, sequenceList_test, climateList_test