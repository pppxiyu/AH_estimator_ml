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

from utils.feature import add_lags
from utils.preprocess import *
from utils.preprocessVis import *


# Single building-weather pair for preliminary analysis

# to import one prototype-weather pair
dataFull = pd.concat([importRawData('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv/MultiFamily-2004____120.csv',
                                    'Environment:Site Total Zone Exhaust Air Heat Loss [J](Hourly)',
                                    ),
                      importWeatherData('./data/weather input', '120'),
                      importTypical('./data/testrun', 'MultiFamily-2004',
                                    'Environment:Site Total Zone Exhaust Air Heat Loss [J](Hourly)') # for adding the typical
                     ],
                     axis = 1)
dataFull.insert(0, 'Climate', dataFull.pop('Climate'))
dataFull['Environment:Site Total Zone Exhaust Air Heat Loss [J](Hourly)'].plot()
plt.show()

# # pac
# plt.figure()
# plot_pacf(dataFull['SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)'].values, lags = 72, method = 'ywm')
# plt.title('Partial Autocorrelation Plot')
# plt.show()

# # ac
# plt.figure()
# plot_acf(dataFull['SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)'].values, lags = 72)
# plt.title('Autocorrelation Plot')
# plt.show()

# # # pac (interactive)
# plotPACF(pacf(dataFull['SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)'].values, nlags = 400, alpha = 0.05), save = './figs/partial_correlation_heatRejection.html')

# # partial cross correlation
# plotPCC(dataFull,
#         ['GLW', 'PSFC', 'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD', 'WINDS'],
#         'SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)',
#         192,
#         save = './figs/partial_cross_correlation_MultiFamility-2013_heatRejection.html'

#         )

# # peaks
# plotPeaks(dataFull[(dataFull['Date/Time'] < '2018-07-31 23:59:59') & (dataFull['Date/Time'] > '2018-07-14 00:00:00')],
#           target = 'Electricity:Facility [J](Hourly)',
#          )


# # Weather feature analysis
# weatherDf = getAllClimatesData(getAllClimates('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
#                                './data/weather input')
# pearsonCorr = weatherDf[['Climate', 'RH']].pivot(columns = 'Climate', values = 'RH').corr(method = 'pearson')
# pearsonCorr.to_csv('./saved/climateCorr/climateCorr.csv')

