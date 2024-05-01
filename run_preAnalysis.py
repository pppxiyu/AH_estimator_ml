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
from utils.vis import *

mode = 'single proto-weather pair'

if mode == 'single proto-weather pair':
    # Single building-weather pair for preliminary analysis
    # to import one prototype-weather pair
    dataFull = pd.concat([importRawData(
        './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv/SingleFamily-2004____36.csv',
        'Electricity:Facility [J](Hourly)',
    ),
        importWeatherData('./data/weather input/2018', '36'),
        importTypical('./data/testrun', 'SingleFamily-2004',
                      'Electricity:Facility [J](Hourly)', 1)
    ],
        axis=1)
    dataFull.insert(0, 'Climate', dataFull.pop('Climate'))
    # dataFull['Electricity:Facility [J](Hourly)'].plot()
    # plt.show()

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

    # partial cross correlation
    # plotPCC(dataFull,
    #         ['GLW', 'PSFC', 'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD', 'WINDS',
    #          'Typical-Electricity:Facility [J](Hourly)',
    #          ],
    #         'Electricity:Facility [J](Hourly)',
    #         192,
    #         save = './figs/partial_cross_correlation_HeavyManufacturing-90_1-2004-ASHRAE 169-2013-3B_36_energyElec.html',
    #         # reverse = True,
    #         # xaxisTitle = 'Advances',
    #         )

    # line chart of ground truth and typical values
    prototypeSelected = 'SingleFamily-2004____36'
    dataFull['DateTime'] = dataFull['Date/Time'].apply(lambda x: str(x).replace('2001', '2018'))
    dataFull = dataFull.iloc[700: 1440]
    plotPrototypeLevelLines_typical(dataFull,
                                    './paper/figs/' + 'line_typical_' + prototypeSelected + '.html',
                                    'Electricity:Facility [J](Hourly)',
                                    prototypeSelected.split('_')[0] + '  ' + prototypeSelected.split('_')[-1]
                                    )

    # # peaks
    # plotPeaks(dataFull[(dataFull['Date/Time'] < '2001-07-31 23:59:59') & (dataFull['Date/Time'] > '2001-07-14 00:00:00')],
    #           target = 'Electricity:Facility [J](Hourly)',
    #          )

# Weather feature analysis
# weatherDf2018 = getAllClimatesData(getAllClimates('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
#                                    './data/weather input/2018')
# weatherDf2018 = weatherDf2018[['Climate', 'SWDOWN']]
# weatherDf2018 = weatherDf2018.pivot(columns='Climate', values='SWDOWN').reset_index(drop=True)
# weatherDf2018 = weatherDf2018.rename(columns = {'SWDOWN': 'SWDOWN_2018'})
#
# weatherDf2016 = getAllClimatesData(getAllClimates('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
#                                    './data/weather input/2016')
# weatherDf2016 = weatherDf2016[['Climate', 'SWDOWN']]
# weatherDf2016 = weatherDf2016.pivot(columns='Climate', values='SWDOWN').reset_index(drop=True)
# weatherDf2016 = weatherDf2016.rename(columns = {'SWDOWN': 'SWDOWN_2016'})
#
# weatherDf = pd.concat([weatherDf2018, weatherDf2016], axis = 1)
# weatherDf = weatherDf.sort_index(axis = 1)


print()
