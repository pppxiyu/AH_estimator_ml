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

# import one prototype with all weather
prototypeName = getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv')[8]
dataFull = getAllData4Prototype(prototypeName,
                                getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv',
                                './data/weather input',
                                './data/testrun',
                                # 'Electricity:Facility [J](Hourly)',
                                'SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)',
                               )
# build features
lagList = (
    (np.arange(24) + 1).tolist()
)
# lagList = [1] + [i + 1 for i in lagList] # add this line if 2 timestamp forward will be predicted
train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                                                                 dataFull,
                                                                                 lagList,
                                                                                 # 'Electricity:Facility [J](Hourly)',
                                                                                 'SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)',
                                                                                 ['GLW', 'PSFC',
                                                                                  'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD',
                                                                                  'WINDS',
                                                                                  # 'Typical-Electricity:Facility [J](Hourly)',
                                                                                  'Typical-SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)',
                                                                                 ],
                                                                                 splitData_biRNN,
                                                                                )



# train with tuning
tuner_biLSTM = kt.BayesianOptimization(
    biLSTM_tuner(),
    objective = "val_loss",
    max_trials = 2,
    overwrite = True,
    directory = "./tuner",
    project_name = 'biLSTM ' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
)

tuner_biLSTM.search(train_X, train_Y,
             epochs = 500,
             callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience = 3, min_delta = 0,),
#                           tf.keras.callbacks.TensorBoard("tuner/biLSTM_tbLogs")
                         ],
             validation_data = (val_X, val_Y),
            )

prediction_list = biLSTM_predict(testSequenceList, tuner_biLSTM)


# metrics
dfEval = organizePredictTrue_biLSTM(prediction_list, testSequenceList, testClimateList)

print('RMSE is:', mean_squared_error(dfEval.true.values, dfEval.predict.values, squared = False))
print('MAE is: ', mean_absolute_error(dfEval.true.values, dfEval.predict.values))
print('MAPE is: ', mean_absolute_percentage_error(dfEval.true.values, dfEval.predict.values))
print('SMAPE is: ', symmetric_mean_absolute_percentage_error(dfEval.true.values, dfEval.predict.values))
# print('CVRMSE is: ', cv_root_mean_squared_error(dfEval.true.values, dfEval.predict.values))
print('CVMAE is: ', cv_mean_absolute_error(dfEval.true.values, dfEval.predict.values))

peaks_true_index = find_peaks(dfEval.true, prominence = 1)[0]
peaks_true_mag = dfEval.true.values[peaks_true_index]
peaks_predict_mag = dfEval.predict.values[peaks_true_index]
print('RMSE at peaks is:', mean_squared_error(peaks_true_mag, peaks_predict_mag, squared = False))
print('MAE at peaks is: ', mean_absolute_error(peaks_true_mag, peaks_predict_mag))
print('SMAPE at peaks is: ', symmetric_mean_absolute_percentage_error(peaks_true_mag, peaks_predict_mag))
# print('CVRMSE is: ', cv_root_mean_squared_error(peaks_true_mag, peaks_predict_mag))
print('CVMAE at peaks is: ', cv_mean_absolute_error(peaks_true_mag, peaks_predict_mag))

peaks_predict_index = find_peaks(dfEval.predict, prominence = 1)[0]
print('Percentage of correct peak timing is: ', (np.intersect1d(peaks_predict_index, peaks_true_index).shape[0] / peaks_true_index.shape[0]))
