from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import pandas as pd
import numpy as np

def organizePredictTrue(predictionList, sequenceList_test, climateList_test, maxLag):
#     sequenceLen = sequenceList_test[0].shape[1]
#     dfTemplate = pd.DataFrame(pd.date_range(start = '2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq = 'H'),
#                  columns = ['DateTime']).iloc[: -(sequenceLen - 1)]

    dfTemplate = pd.DataFrame(pd.date_range(start = '2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq = 'H'),
                 columns = ['DateTime']).iloc[: -maxLag]
    dfList = []
    for prediction, sequence_test, climate in zip(predictionList, sequenceList_test, climateList_test):
        predict = prediction[:, 0]
        true = sequence_test[:, -1, 0]
#         true = sequence_test[:, -2, 0] # 2 timestamps forward

        df = dfTemplate.copy()
        df['climateID'] = [climate] * len(df)
        df['predict'] = predict
        df['true'] = true
        dfList.append(df)

    dfFull = pd.concat(dfList, axis = 0)
    return dfFull

def organizePredictTrue_biLSTM(predictionList, sequenceList_test, climateList_test):
    sequenceLen = sequenceList_test[0].shape[1]
    dfTemplate = pd.DataFrame(pd.date_range(start = '2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq = 'H'),
                 columns = ['DateTime']).iloc[((sequenceLen - 1) // 2): -((sequenceLen - 1) // 2)]
    dfList = []
    for prediction, sequence_test, climate in zip(predictionList, sequenceList_test, climateList_test):
        predict = prediction[:, 0]
        true = sequence_test[:, ((sequenceLen - 1) // 2): ((sequenceLen + 1) // 2), 0]
#         true = sequence_test[:, -2, 0] # 2 timestamps forward

        df = dfTemplate.copy()
        df['climateID'] = [climate] * len(df)
        df['predict'] = predict
        df['true'] = true
        dfList.append(df)

    dfFull = pd.concat(dfList, axis = 0)
    return dfFull

def organizePredictTrue_linear(predictionList, sequenceList_test, climateList_test, maxLag):

    dfTemplate = pd.DataFrame(pd.date_range(start = '2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq = 'H'),
                 columns = ['DateTime']).iloc[: -maxLag]
    dfList = []

    for prediction, sequence_test, climate in zip(predictionList, sequenceList_test, climateList_test):
        predict = prediction[:, 0]
        true = sequence_test[:, -1, 0]
        true = true.reshape(true.shape[0], -1)

        df = dfTemplate.copy()
        df['climateID'] = [climate] * len(df)
        df['predict'] = predict
        df['true'] = true
        dfList.append(df)

    dfFull = pd.concat(dfList, axis = 0)
    return dfFull

def organizePredictTrue_mlp(predictionList, sequenceList_test, climateList_test, maxLag):

    dfTemplate = pd.DataFrame(pd.date_range(start = '2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq = 'H'),
                 columns = ['DateTime']).iloc[: -maxLag]
    dfList = []

    for prediction, sequence_test, climate in zip(predictionList, sequenceList_test, climateList_test):
        true = sequence_test[:, -1, 0]
        true = true.reshape(true.shape[0], -1)

        df = dfTemplate.copy()
        df['climateID'] = [climate] * len(df)
        df['predict'] = prediction
        df['true'] = true
        dfList.append(df)

    dfFull = pd.concat(dfList, axis = 0)
    return dfFull

organizePredictTrue_xgboost = organizePredictTrue_linear
    
def symmetric_mean_absolute_percentage_error(true, predict):
    return np.mean(
        np.abs(predict - true) /
        ((predict + true) / 2)
    )

def cv_mean_absolute_error(true, predict):
    return mean_absolute_error(true, predict) / np.mean(true)

def cv_root_mean_squared_error(true, predict):
    return mean_squared_error(true, predict, squared = False) / np.mean(true)

def cv_mean_absolute_error_wAbs(true, predict):
    return mean_absolute_error(true, predict) / np.mean(np.abs(true))

def cv_root_mean_squared_error_wAbs(true, predict):
    return mean_squared_error(true, predict, squared = False) / np.mean(np.abs(true))

def coolingElec(dfEval, climateList_test, timeRange, grid, locs, saveAddr):
    if timeRange:
        dfEval = dfEval.loc[dfEval.DateTime > timeRange[0]]
        dfEval = dfEval.loc[dfEval.DateTime < timeRange[1]]

    maeDict = {}
    for cli in climateList_test:
        dfSelect_0 = dfEval[dfEval.climateID == cli]
        mae = mean_absolute_error(dfSelect_0.true.values, dfSelect_0.predict.values)
        maeDict[cli] = mae

    fig = make_subplots(
        rows = grid[0], cols = grid[1],
        subplot_titles = ['Climate ID: ' + climateID + ' (MAE: '
                          + str(maeDict[climateID]) + ')'
                          for climateID in climateList_test]
    )

    for cli, loc in zip(climateList_test, locs):
        dfSelect = dfEval[dfEval.climateID == cli]
        x = dfSelect.DateTime
        predict = dfSelect.predict
        true = dfSelect.true
        fig.add_trace(go.Scatter(x = x, y = predict,
                                 mode = 'lines', name = 'Prediction: ' + cli,
                                 line = dict(color = '#3182bd')),
                      row = loc[0], col = loc[1])
        fig.add_trace(go.Scatter(x = x, y = true,
                                 mode = 'lines', name = 'Ground Truth: ' + cli,
                                 line = dict(color = '#feb24c')),
                      row = loc[0], col = loc[1])
        fig.update_xaxes(title_text = "Time (Hour)", row = loc[0], col = loc[1])
        fig.update_yaxes(title_text = "Cooling Electricity (KWH)", row = loc[0], col = loc[1])

    fig.write_html(saveAddr)