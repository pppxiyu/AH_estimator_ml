from sklearn.metrics import mean_absolute_percentage_error

import pandas as pd
import numpy as np

import math
import os
import re

from utils.eval import *
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.signal import find_peaks
import json

import plotly.express as px
import plotly.graph_objects as go


## sampling ####################
def splitBuildingWeatherPair(addr):
    # USE: randomly select some weather zones for test
    #      make sure the training pairs have all building types
    #      return pairs for training and test
    # INPUT: building meta data csv
    # OUTPUT: lists of tuples, each tuple has building type and weather ID

    # building stock info for all census tracts
    buildingMeta = pd.read_csv(addr)
    #     buildingMeta_grouped = buildingMeta.groupby(['id.tract', 'idf.kw', 'id.grid.coarse'])['building.area.m2'].sum()
    buildingMeta_grouped_pair = buildingMeta.groupby(['idf.kw', 'id.grid.coarse']).count()

    # split train and test config
    climates = buildingMeta_grouped_pair.index.get_level_values(1).unique().values
    num_climate = buildingMeta_grouped_pair.index.get_level_values(1).nunique()
    indices = np.random.choice(climates.size, size=math.floor(num_climate * 0.15), replace=False)
    mask = np.ones(climates.size, dtype=bool)
    mask[indices] = False
    climates_select = climates[indices]  # select climates for test
    climates_unSelect = climates[mask]  # Get the unselected elements

    # training data pairs
    mask = buildingMeta_grouped_pair.index.get_level_values(1).isin(climates_unSelect)
    originalBuildingTypeNum = buildingMeta_grouped_pair.index.get_level_values(0).nunique()
    if buildingMeta_grouped_pair[mask].index.get_level_values(0).nunique() != originalBuildingTypeNum:
        raise ValueError("Some building type is missing in training dataset.")
    pairList_train = buildingMeta_grouped_pair.loc[mask].index.tolist()

    # test data pairs
    mask = buildingMeta_grouped_pair.index.get_level_values(1).isin(climates_select)
    pairList_test = buildingMeta_grouped_pair.loc[mask].index.tolist()

    return pairList_train, pairList_test


## normalize to per m2 #########################
def getBuildingArea_prototype(addr, verbose = 0):
    # USE: get the building area of building prototype
    # INPUT: the address of simulation outputs directory
    # OUTPUT: dict

    lookupLog = []
    buildingArea_dict = {}
    for name in os.listdir(addr):
        if name != '.DS_Store':
            proto = name.split('____')[0]

            # only lookup once for each prototype
            if (proto in lookupLog) == False:
                print(f'Found proto: {proto}')
                lookupLog.append(proto)

                # look up the building area of this prototype
                with open(addr + '/' + name + '/eplustbl.htm', "r") as file:
                    lines = file.readlines()
                    for line, index in zip(lines, range(len(lines))):
                        if '<td align="right">Total Building Area</td>' in line:
                            buildingArea = re.search('[0-9]+\.[0-9]+', lines[index + 1]).group()
                            if verbose == 1:
                                print('Get the building area line: ')
                                print(line, lines[index + 1].strip())
                                print('Get building area: ')
                                print(proto, ': ', buildingArea)
                                print()
                            buildingArea_dict[proto] = float(buildingArea)
    return buildingArea_dict

def normalize_perM2(predictionDict, pairList_test, buildingArea_dict):
    # USE: normalize estimations for all prototype-weather pair in test set using building area
    # OUTPUT: df, similar to output of trainPredict_allPairs_LSTM, with two more normalized columns

    predictionDict_norm = {}
    for testPair in pairList_test:
        # get the estimates
        testPairName = testPair[0] + '____' + str(testPair[1])
        testPair_df = predictionDict[testPairName]
        # get building area
        area = buildingArea_dict[testPair[0]]
        # normalize
        testPair_df['estimatePerM2'] = testPair_df['estimate'] / area
        testPair_df['truePerM2'] = testPair_df['true'] / area
        # save to a new dict
        predictionDict_norm[testPairName] = testPair_df
    return predictionDict_norm


#### scale up to tracts ######################
def getBuildingArea_tracts(addr, pairList_test):
    # USE: get the building area of each prototype-weather pair for all tracts
    # INPUT: building meta data csv file
    # OUTPUT: df, with tract, building type, weather column, and the corresponding building area

    buildingMeta = pd.read_csv(addr)
#     pairList_test_idfkw = [item[0] for item in pairList_test] # building prototype
    pairList_test_idgridcoarse = [item[1] for item in pairList_test] # weather
#     buildingMeta_forTest = buildingMeta[(buildingMeta['idf.kw'].isin(pairList_test_idfkw)) & (buildingMeta['id.grid.coarse'].isin(pairList_test_idgridcoarse))]
    buildingMeta_forTest = buildingMeta[buildingMeta['id.grid.coarse'].isin(pairList_test_idgridcoarse)]
    buildingMeta_forTest_tract = buildingMeta_forTest.groupby(['id.tract', 'idf.kw', 'id.grid.coarse'])['building.area.m2'].sum()
    buildingMeta_forTest_tract = buildingMeta_forTest_tract.reset_index()
    return buildingMeta_forTest_tract

def predict_tracts(predictionDict_norm, buildingMeta_tract):
    # USE: scale up the normed estimation of building-weather pairs to tracts
    # INPUT: normed estimation of building-weather pairs; building area of each prototype-weather pair for all tracts
    # OUTPUT: df, estimation for tracts

    estimate_tract_df_list = []
    for tract in buildingMeta_tract['id.tract'].unique().tolist():  # iterate all tracts

        tractDf = buildingMeta_tract[buildingMeta_tract['id.tract'] == tract] # get all building-weather pair in the tract
        estimateTract = np.zeros(len(list(predictionDict_norm.values())[0])) # initiate estimate
        for i in range(len(tractDf)): # loop through all the building-weather pairs in this tract
            tractDf_row = tractDf.iloc[i]
            pairName = tractDf_row['idf.kw'] + '____' + str(tractDf_row['id.grid.coarse'])
            estimateTract += (predictionDict_norm[pairName]['estimatePerM2'] * tractDf_row['building.area.m2']).values
        # record it into df
        estimate_tract_df = list(predictionDict_norm.values())[0].loc[:, ['DateTime']]
        estimate_tract_df['estimate'] = estimateTract
        estimate_tract_df.insert(0, 'geoid', [tract] * len(estimate_tract_df))
        estimate_tract_df_list.append(estimate_tract_df)
    # concat df for all tracts
    estimate_tract_df_all = pd.concat(estimate_tract_df_list, axis = 0, ignore_index = True)
    return estimate_tract_df_all

def getTracts_remove(addr, tractsMeta):
    # USE: get the names of tracts that having weather not in test set
    # INPUT: addr, the address of building meta data
    #        tractsMeta, tracts meta data, inclduing the weathers in the df
    # OUTPUT: a list of tracts names that should be removed

    # get original weathers in tracts
    buildingMeta = pd.read_csv(addr)
    tractList = tractsMeta['id.tract'].unique().tolist()
    tractWeatherOriginal = buildingMeta[buildingMeta['id.tract'].isin(tractList)]
    tractWeatherOriginal = tractWeatherOriginal.groupby(['id.tract', 'id.grid.coarse']).count().reset_index()
    tractWeatherOriginal = tractWeatherOriginal[['id.tract', 'id.grid.coarse']]

    # get current weathers in tracts
    tractsMeta = tractsMeta.groupby(['id.tract', 'id.grid.coarse']).count().reset_index()
    tractsMeta = tractsMeta[['id.tract', 'id.grid.coarse']]

    # get the rows that is in original but not in the current one
    merged = tractWeatherOriginal.merge(tractsMeta, how = 'left', on = ['id.tract', 'id.grid.coarse'], indicator = True)
    originalOnly = merged[merged._merge == 'left_only']
    tractsWithProblem = originalOnly['id.tract'].unique().tolist()

    return tractsWithProblem


## get ground truth
def loadTractData(addr, colName):
    # USE: load the true tract-level data
    tractData = pd.read_csv(addr, usecols = ['geoid', 'timestamp', colName])
    tractData['timestamp'] = np.repeat(pd.date_range(start = '2018-01-01 00:00:00',
                                                     end = '2018-12-31 23:00:00',
                                                     freq = 'H'),
                                       len(tractData) / 8760,)
    tractData[colName] = tractData[colName] / 3.6e+6
    return tractData

def filterTractData(tractData, estimateTractData):
    # USE: filter the true energy data using the geoid in test set
    geoid4test = list(estimateTractData.geoid.unique())
    tractDataFiltered = tractData[tractData.geoid.isin(geoid4test)]
    tractDataFiltered = tractDataFiltered.sort_values(by = ['geoid', 'timestamp'])
    return tractDataFiltered


# eval
def combineEstimateTrue(true, estimate, target):
    # make the datetime index consistent
    true = true[(true.timestamp >= estimate['DateTime'].iloc[0]) & ((true.timestamp <= estimate['DateTime'].iloc[-1]))]
    # combine
    df = true.copy()
    df = df[['geoid', 'timestamp']]
    df['true'] = true[target].to_list()
    df['estimate'] = estimate['estimate'].to_list()
    return df

def reloadPrototypeLevelPred(folder):
    files = [f for f in os.listdir(folder)]
    dfs = {}
    for filename in files:
        df = pd.read_csv(folder + '/' + filename)
        filename_clean = os.path.splitext(filename)[0]
        dfs[filename_clean] = df
    return dfs

def metricPrototypeWeather(prediction_dict, metricFunc):
    # show the error at the building-weather pair level
    metric_dict = {}
    for k in list(prediction_dict.keys()):
        metric= metricFunc(prediction_dict[k].true, prediction_dict[k].estimate)
        metric_dict[k] = metric
    return metric_dict

def metricPrototype(metric_dict, metricName):
    df = pd.DataFrame(list(metric_dict.items()), columns = ['pair', metricName])
    df['prototype'] = df.pair.str.split('____').str[0]
    df_group = df[['prototype', metricName]].groupby('prototype').mean()
    df_group = df_group.reset_index()
    df_group = df_group.sort_values(metricName, ascending = True)
    return df_group


def getTractLevelMetrics(predTractLevel, addr, computeTime = None):

    # USE: get the metrics at tract level
    # INPUT: the folder containing running results
    # OUTPUT: save json to the folder

    metrics = {}

    metrics['RMSE'] = mean_squared_error(predTractLevel.true.values, predTractLevel.estimate.values, squared = False)
    metrics['MAE'] = mean_absolute_error(predTractLevel.true.values, predTractLevel.estimate.values)
    metrics['MAPE'] = mean_absolute_percentage_error(predTractLevel.true.values, predTractLevel.estimate.values)
    metrics['CVRMSE'] = cv_root_mean_squared_error(predTractLevel.true.values, predTractLevel.estimate.values)
    metrics['CVMAE'] = cv_mean_absolute_error(predTractLevel.true.values, predTractLevel.estimate.values)
    metrics['CVRMSE_wAbs'] = cv_root_mean_squared_error_wAbs(predTractLevel.true.values, predTractLevel.estimate.values)
    metrics['CVMAE_wAbs'] = cv_mean_absolute_error_wAbs(predTractLevel.true.values, predTractLevel.estimate.values)

    peaks_true_index = find_peaks(predTractLevel.true, prominence = 1)[0]
    peaks_true_mag = predTractLevel.true.values[peaks_true_index]
    peaks_predict_mag = predTractLevel.estimate.values[peaks_true_index]
    peaks_predict_index = find_peaks(predTractLevel.estimate, prominence = 1)[0]
    metrics['PEAK_RMSE'] =  mean_squared_error(peaks_true_mag, peaks_predict_mag, squared = False)
    metrics['PEAK_MAE'] = mean_absolute_error(peaks_true_mag, peaks_predict_mag)
    metrics['PEAK_CVRMSE'] = cv_root_mean_squared_error(peaks_true_mag, peaks_predict_mag)
    metrics['PEAK_CVMAE'] = cv_mean_absolute_error(peaks_true_mag, peaks_predict_mag)
    metrics['PEAK_CVRMSE_wAbs'] = cv_root_mean_squared_error_wAbs(peaks_true_mag, peaks_predict_mag)
    metrics['PEAK_CVMAE_wAbs'] = cv_mean_absolute_error_wAbs(peaks_true_mag, peaks_predict_mag)
    metrics['PEAK_CorrectTiming'] = (np.intersect1d(peaks_predict_index, peaks_true_index).shape[0] / peaks_true_index.shape[0])

    metrics['exeTIME'] = computeTime

    print('RMSE is:', mean_squared_error(predTractLevel.true.values, predTractLevel.estimate.values, squared = False))
    print('MAE is: ', mean_absolute_error(predTractLevel.true.values, predTractLevel.estimate.values))

    print('MAPE is: ', mean_absolute_percentage_error(predTractLevel.true.values, predTractLevel.estimate.values))
    print('CVRMSE is: ', cv_root_mean_squared_error(predTractLevel.true.values, predTractLevel.estimate.values))
    print('CVMAE is: ', cv_mean_absolute_error(predTractLevel.true.values, predTractLevel.estimate.values))
    print('CVRMSE_wAbs is:', cv_root_mean_squared_error_wAbs(predTractLevel.true.values, predTractLevel.estimate.values))
    print('CVMAE_wAbs is:', cv_mean_absolute_error_wAbs(predTractLevel.true.values, predTractLevel.estimate.values))

    print('RMSE at peaks is:', mean_squared_error(peaks_true_mag, peaks_predict_mag, squared = False))
    print('MAE at peaks is: ', mean_absolute_error(peaks_true_mag, peaks_predict_mag))
    print('CVRMSE at peak is: ', cv_root_mean_squared_error(peaks_true_mag, peaks_predict_mag))
    print('CVMAE at peaks is: ', cv_mean_absolute_error(peaks_true_mag, peaks_predict_mag))
    print('PEAK_CVRMSE_wAbs at peak is', cv_root_mean_squared_error_wAbs(peaks_true_mag, peaks_predict_mag))
    print('PEAK_CVMAE_wAbs at peak is', cv_mean_absolute_error_wAbs(peaks_true_mag, peaks_predict_mag))
    print('Percentage of correct peak timing is: ', (np.intersect1d(peaks_predict_index, peaks_true_index).shape[0] / peaks_true_index.shape[0]))

    print('exeTIME is: ', computeTime)

    with open(addr + '/' + 'tractLevelMetrics.json', 'w') as f:
        json.dump(metrics, f)

    print('Tract level metrics saved.')

    return 

def getPrototypeLevelMetrics(predPrototypeLevel, addr):
    # USE: get the metrics at prototype level
    # INPUT: the folder containing running results
    # OUTPUT: save json to the folder

    metrics = {}
    metrics['MAPE'] = metricPrototype(metricPrototypeWeather(predPrototypeLevel, mean_absolute_percentage_error), 'MAPE')

    with open(addr + '/' + 'prototypeLevelMetrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return 

def plotPrototypeLevelMetrics(predPrototypeLevel, addr, metricFunc, metricName):
    # USE: draw the metrics plot at prototype level
    # INPUT: the folder containing running results
    # OUTPUT: save plot to the folder

    metric_prototype_ave_dict = metricPrototype(metricPrototypeWeather(predPrototypeLevel, metricFunc), metricName)
    metric_prototype_ave_dict.plot.barh(x='prototype', y=metricName, figsize=(10, 10))
    plt.subplots_adjust(left=0.55)
    plt.savefig(addr + '/prototypeLevel' + metricName + '.png')

    print('Prototype level metric fig saved.')

    return 

def plotTractLevelMetrics(predTractLevel, addr, metricFunc, metricName):
    # USE: draw the metrics plot at tract level
    # INPUT: the folder containing running results
    # OUTPUT: save plot to the folder

    metricByTracts = predTractLevel.groupby('geoid').apply(lambda g: metricFunc(g.true, g.estimate))
    metricByTracts.to_frame().hist(bins=50, grid=False, color='#607c8e')
    plt.xlabel(metricName)
    plt.savefig(addr + '/tractLevel' + metricName + '.png')

    return

def plotTractLevelPredictionLine(df, tractSelect, addr):
    # USE: draw the prediction line graph for one census tract
    # INPUT: the tract level df, containing both true and estimate
    #        census tract name
    #        the dir of the experiment folder
    # OUTPUT: save plot to the folder

    dfSelect = df[df.geoid == tractSelect]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dfSelect.timestamp,
                             y = dfSelect.true,
                             name = "true",
                             line_shape = 'linear'))
    fig.add_trace(go.Scatter(x = dfSelect.timestamp,
                             y = dfSelect.estimate,
                             name = "estimate",
                             line_shape = 'linear'))
    fig.write_html(addr + '/' + 'tractEstimation_' + str(tractSelect) + '.html')

    return