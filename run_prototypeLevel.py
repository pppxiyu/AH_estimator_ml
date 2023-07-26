import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import find_peaks

import os
import math
import random
import warnings
import re
from datetime import datetime

from utils.feature import *
from utils.preprocess import *
from utils.preprocessVis import *
from utils.eval import *

from models.LSTM import *
from models.biLSTM import *
from models.mlp import *
from models.linear import *
from models.xgboost import *

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier

prototypeName = getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv')[28]
target = 'Electricity:Facility [J](Hourly)'
features = ['GLW', 'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD', 'Typical-Electricity:Facility [J](Hourly)']
lagList = (
    (np.arange(24) + 1).tolist()
)
tuneTrails = 1
trainEpoch = 1
modelName = 'mlp'

# import one prototype with all weather
dataFull = getAllData4Prototype(prototypeName,
                                getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv',
                                './data/weather input',
                                './data/testrun',
                                target,
                               )
if modelName == 'LSTM':
    # build features
    train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                                                                     dataFull,
                                                                                     lagList,
                                                                                     target,
                                                                                     features,
                                                                                     splitData,
                                                                                    )
    # # without tuning
    # prediction_list = LSTM_train_predict(0.001, 512, 512, train_X, train_Y, val_X, val_Y, testSequenceList)
    # train with tuning
    tuner_LSTM = kt.BayesianOptimization(
        LSTM_tuner(),
        objective = "val_loss",
        max_trials = tuneTrails,
        overwrite = True,
        directory = "./tuner",
        project_name = 'LSTM ' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
    )
    tuner_LSTM.search(train_X, train_Y,
                 epochs = trainEpoch,
                 callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience = 3, min_delta = 0,),
    #                           tf.keras.callbacks.TensorBoard("tuner/LSTM_tbLogs")
                             ],
                 validation_data = (val_X, val_Y),
                )
    prediction_list = LSTM_predict(testSequenceList, tuner = tuner_LSTM)
    dfEval = organizePredictTrue(prediction_list, testSequenceList, testClimateList, lagList[-1])

if modelName == 'biLSTM':
    train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(
        getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
        dataFull,
        lagList,
        target,
        features,
        splitData_biRNN,
        )
    # train with tuning
    tuner_biLSTM = kt.BayesianOptimization(
        biLSTM_tuner(),
        objective="val_loss",
        max_trials=tuneTrails,
        overwrite=True,
        directory="./tuner",
        project_name='biLSTM ' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    tuner_biLSTM.search(train_X, train_Y,
                        epochs=trainEpoch,
                        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3, min_delta=0, ),
                                   #                           tf.keras.callbacks.TensorBoard("tuner/biLSTM_tbLogs")
                                   ],
                        validation_data=(val_X, val_Y),
                        )
    prediction_list = biLSTM_predict(testSequenceList, tuner_biLSTM)
    dfEval = organizePredictTrue_biLSTM(prediction_list, testSequenceList, testClimateList)

if modelName == 'linear':
    train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(
        getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
        dataFull,
        lagList,
        target,
        features,
        splitData,
        )
    # cancel the temporal dimension and de-stack the features of lagged timestamps
    train_X = train_X.reshape(train_X.shape[0], -1)
    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)
    val_Y = val_Y.reshape(val_Y.shape[0], -1)
    print('Updated train_X shape is: ', train_X.shape)
    print('Updated train_Y shape is: ', train_Y.shape)
    # train
    model = LinearRegression()
    model.fit(train_X, train_Y)
    # val
    val_predY = model.predict(val_X)
    print('CVMAE_wAbs using val data is:', cv_mean_absolute_error_wAbs(val_Y, val_predY))
    # pred and eval
    prediction_list = linear_predict(testSequenceList, model)
    dfEval = organizePredictTrue_linear(prediction_list, testSequenceList, testClimateList, lagList[-1])

if modelName == 'mlp':
    train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(
        getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
        dataFull,
        lagList,
        target,
        features,
        splitData,
        )
    # cancel the temporal dimension and de-stack the features of lagged timestamps
    train_X = train_X.reshape(train_X.shape[0], -1)
    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)
    val_Y = val_Y.reshape(val_Y.shape[0], -1)
    print('Updated train_X shape is: ', train_X.shape)
    print('Updated train_Y shape is: ', train_Y.shape)
    # train
    model = MLPRegressor(
        hidden_layer_sizes=(100, 75, 50),
        early_stopping=True,
        n_iter_no_change=3,
        validation_fraction=0.15,
        learning_rate_init=0.001,
    )
    model.fit(train_X, train_Y)
    # val
    val_predY = model.predict(val_X)
    print('CVMAE_wAbs using the val data is: ', cv_mean_absolute_error_wAbs(val_Y, val_predY))
    # pred and eval
    prediction_list = mlp_predict(testSequenceList, model)
    dfEval = organizePredictTrue_mlp(prediction_list, testSequenceList, testClimateList, lagList[-1])

if modelName == 'xgboost':
    train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(
        getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
        dataFull,
        lagList,
        target,
        features,
        splitData,
        )
    # cancel the temporal dimension and de-stack the features of lagged timestamps
    train_X = train_X.reshape(train_X.shape[0], -1)
    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)
    val_Y = val_Y.reshape(val_Y.shape[0], -1)
    print('Updated train_X shape is: ', train_X.shape)
    print('Updated train_Y shape is: ', train_Y.shape)
    # train
    model = XGBClassifier(
        n_estimators=100,
        subsample=0.5,
        max_depth=6,
        eta=0.001,
        gamma=0.0001,
        reg_alpha=0,
        reg_lambda=1,
    )
    model.fit(train_X, train_Y)
    # val
    val_predY = model.predict(val_X)
    print('CVMAE_wAbs using the val data is: ', cv_mean_absolute_error_wAbs(val_Y, val_predY))
    # pred and eval
    prediction_list = xgboost_predict(testSequenceList, model)
    dfEval = organizePredictTrue_xgboost(prediction_list, testSequenceList, testClimateList, lagList[-1])


# metrics
print('RMSE is:', mean_squared_error(dfEval.true.values, dfEval.predict.values, squared = False))
print('MAE is: ', mean_absolute_error(dfEval.true.values, dfEval.predict.values))
print('MAPE is: ', mean_absolute_percentage_error(dfEval.true.values, dfEval.predict.values))

peaks_true_index = find_peaks(dfEval.true, prominence = 1)[0]
peaks_true_mag = dfEval.true.values[peaks_true_index]
peaks_predict_mag = dfEval.predict.values[peaks_true_index]
print('RMSE at peaks is:', mean_squared_error(peaks_true_mag, peaks_predict_mag, squared = False))
print('MAE at peaks is: ', mean_absolute_error(peaks_true_mag, peaks_predict_mag))

peaks_predict_index = find_peaks(dfEval.predict, prominence = 1)[0]
print('Percentage of correct peak timing is: ', (np.intersect1d(peaks_predict_index, peaks_true_index).shape[0] / peaks_true_index.shape[0]))

# # vis
# coolingElec(dfEval, testClimateList,
#             ('2018-01-01 00:00:00', '2018-12-31 23:00:00'),
#             (1, 1),
#             [(1, 1)],
#             './figs/test_0101-1231_College-90_1-2004-ASHRAE 169-2013-3B.html')

