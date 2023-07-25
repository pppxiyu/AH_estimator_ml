import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import find_peaks

from utils.feature import *
from utils.preprocess import *
from utils.eval import *

from models.mlp import *

from sklearn.neural_network import MLPRegressor

# import one prototype with all weather
prototypeName = getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv')[28]
dataFull = getAllData4Prototype(prototypeName,
                                getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv',
                                './data/weather input',
                                './data/testrun',
                                'Electricity:Facility [J](Hourly)',
                               )

# build features
lagList = (
    (np.arange(24) + 1).tolist()
)
train_X, train_Y, val_X, val_Y, testSequenceList, testClimateList = makeDatasets(getClimateName4Prototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv', prototypeName),
                                                                                 dataFull,
                                                                                 lagList,
                                                                                 'Electricity:Facility [J](Hourly)',
                                                                                 ['GLW', 'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD',
                                                                                  'Typical-Electricity:Facility [J](Hourly)'
                                                                                 ],
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
    hidden_layer_sizes = (100, 75, 50),
    early_stopping = True,
    n_iter_no_change = 3,
    validation_fraction = 0.15,
    learning_rate_init = 0.001,
    )
model.fit(train_X, train_Y)

# val
val_predY = model.predict(val_X)
cv_mean_absolute_error(val_Y, val_predY)

# pred and eval
prediction_list = mlp_predict(testSequenceList, model)

dfEval = organizePredictTrue_mlp(prediction_list, testSequenceList, testClimateList, lagList[-1])

print('RMSE is:', mean_squared_error(dfEval.true.values, dfEval.predict.values, squared = False))
print('MAE is: ', mean_absolute_error(dfEval.true.values, dfEval.predict.values))
print('MAPE is: ', mean_absolute_percentage_error(dfEval.true.values, dfEval.predict.values))
print('CVRMSE is: ', cv_root_mean_squared_error(dfEval.true.values, dfEval.predict.values))
print('CVMAE is: ', cv_mean_absolute_error(dfEval.true.values, dfEval.predict.values))

peaks_true_index = find_peaks(dfEval.true, prominence = 1)[0]
peaks_true_mag = dfEval.true.values[peaks_true_index]
peaks_predict_mag = dfEval.predict.values[peaks_true_index]
print('RMSE at peaks is:', mean_squared_error(peaks_true_mag, peaks_predict_mag, squared = False))
print('MAE at peaks is: ', mean_absolute_error(peaks_true_mag, peaks_predict_mag))

peaks_predict_index = find_peaks(dfEval.predict, prominence = 1)[0]
print('Percentage of correct peak timing is: ', (np.intersect1d(peaks_predict_index, peaks_true_index).shape[0] / peaks_true_index.shape[0]))

