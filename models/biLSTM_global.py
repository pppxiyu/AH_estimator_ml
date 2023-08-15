import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from utils.feature import *
from utils.preprocess import *
from datetime import datetime

import pandas as pd
import numpy as np

from models.biLSTM import *

import sys


def getPrototypeWeights(addr):
    # USE: calculate the average building area in all weather zone for each prototype
    # INPUT: addr, string, the dir of building meta data
    # OUTPUT: df, weights of each prototype
    buildingMeta = pd.read_csv(addr)
    prototypeWeatherArea = buildingMeta.groupby(['idf.kw', 'id.grid.coarse'])['building.area.m2'].sum().reset_index()
    prototypeAveArea = prototypeWeatherArea.groupby('idf.kw')[['building.area.m2']].mean()
    prototypeAveArea['weight'] = prototypeAveArea['building.area.m2'] / prototypeAveArea['building.area.m2'].sum()
    return prototypeAveArea[['weight']]

class biLSTM_global(tf.keras.Model):
    def __init__(self, hp_units, name=None, **kwargs):
        super().__init__(**kwargs)
        self.hp_units = hp_units
        self.norm = keras.layers.BatchNormalization()
        self.lstm0 = keras.layers.LSTM(self.hp_units, return_sequences=True)
        self.bilstm0 = keras.layers.Bidirectional(self.lstm0)
        self.lstm1 = keras.layers.LSTM(self.hp_units, return_sequences=True)
        self.bilstm1 = keras.layers.Bidirectional(self.lstm1)
        self.lstm2 = keras.layers.LSTM(512, return_sequences=True)
        self.bilstm2 = keras.layers.Bidirectional(self.lstm2)
        self.dense0 = keras.layers.Dense(385)
        self.dense1 = keras.layers.Dense(256)
        self.dense2 = keras.layers.Dense(128)
        self.dense3 = keras.layers.Dense(1)

    def build(self, input_shape):
        self.sequneceLen = input_shape[1]
        self.featureNum = input_shape[2]
        self.midLoc = (self.sequneceLen + 1) // 2

    def call(self, x):
        x = self.norm(x)
        x = self.bilstm0(x)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = x[:, (self.midLoc - 1): self.midLoc, :]
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x_out = self.dense3(x)
        return x_out

class biLSTM_tuner_global(kt.HyperModel):
    def build(self, hp):
        hp_units = hp.Int('units', min_value = 768, max_value = 768, step = 128)
        model = biLSTM_global(hp_units)
        # hp_lr = hp.Float('learningRate', 0.0005, 0.001, step = 0.0005)
        hp_lr = 0.001
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_lr),
                      loss = "mse",
                      )
        return model

    def fit(self, hp, model, *args, **kwargs):
        hp_batchSize = hp.Choice('batchSize', values = [
            # 512,
            # 768,
            # 1024,
            2048,
        ])
        return model.fit(
            *args,
            batch_size = hp_batchSize,
            **kwargs,
        )

def train_tract_biRNN_global(dirs, pairList_train, pairList_test, featureList, target, lag, tuneTrail, maxEpoch, metaDataDir, randomSeed, dayOfWeekJan1):
    # USE: use the building-weather pairs in the train pair set to train
    #      do prediction using the new weathers in the test pair set
    # INPUT: all prototype list, pairs for train, pairs for test, featrue names, target name, lag list, ifTune True or False
    # OUTPUT: dict, each value is the prediction for a pair in the test pair set

    dirEnergy = dirs[0]
    dirWeather = dirs[1]
    dirTypical = dirs[2]
    try:
        dirEnergyTarget = dirs[3]
        dirWeatherTarget = dirs[4]
        dirTypicalTarget = dirs[5]
    except:
        dirEnergyTarget = dirEnergy
        dirWeatherTarget = dirWeather
        dirTypicalTarget = dirTypical
        print('Evaluation mode. Train and test data are in same year.')

    # protoList = getAllPrototype(dirEnergy)
    protoList = sorted(list(set([item[0] for item in pairList_test])))

    ###################### Merging data for all prototype and ecoding feature ##################
    try: # check if there is saved data
        data = pd.read_csv('./saved/encoded/randomSeed_' + str(randomSeed) + '/data.csv')
        data['Climate'] = data['Climate'].astype(str)
        floatCol = data.select_dtypes(include=['float64']).columns  # change float64 to 32
        data[floatCol] = data[floatCol].astype(np.float32)
        print('Saved data is used.')

    except: # if not, process data
        print('Saved data not found, create new one.')
        dfList = []
        # loop
        for prototypeSelect in protoList:

            print()
            print('---------- Merging data: ', prototypeSelect, ' ----------')

            # get weathers names in train_pairs for the prototype
            protoClimate = [str(item[1]) for item in pairList_train if item[0] == prototypeSelect]
            if len(protoClimate) < 1:
                warnings.warn("Some building type is missing in training dataset.")

            # get weather data in train_pairs for the prototype
            data_1 = getAllData4Prototype(prototypeSelect, protoClimate,
                                          dirEnergy,
                                          dirWeather,
                                          dirTypical,
                                          target,
                                          1, # hard coded, because typical value is obtained in 2018
                                          )
            data_1['prototype'] = prototypeSelect
            dfList.append(data_1)

        data = pd.concat(dfList)
        floatCol = data.select_dtypes(include = ['float64']).columns # change float64 to 32
        data[floatCol] = data[floatCol].astype(np.float32)

        # save
        data.to_csv('./saved/encoded/randomSeed_' + str(randomSeed) + '/data.csv', index = False)

    # targer mean ecoding and keep the encoding info
    encodingDf = data.groupby('prototype').mean()[[target]]
    encodingDf = encodingDf.reset_index().rename(columns = {target: 'targetMean'})
    data_encoded = data.merge(encodingDf, how = 'left', on = 'prototype')
    data_encoded = data_encoded.drop(columns = ['prototype'])

    # # one-hot encoding and keep the encoding info
    # data_encoded = pd.get_dummies(data, columns = ['prototype'])
    # encodingColumns = [col for col in data_encoded.columns if col.startswith('prototype_')]
    # encodingDf = data_encoded[encodingColumns]
    # encodingDf = encodingDf.drop_duplicates()
    # encodingDf = encodingDf.set_index(encodingDf.columns)

    print()
    print('---------- Merged data has been encoded or loaded----------')


    try: # try to get data ready for training
        trainX = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/trainX.npy')
        trainY = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/trainY.npy')
        valX = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/valX.npy')
        valY = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/valY.npy')
        # sampleWeightTrain = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/sampleWeightTrain.npy')
        # sampleWeightVal = np.load('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/sampleWeightVal.npy')

    except:
        ############### Adding sample weight to data #######################

        prototypeWeights = getPrototypeWeights(metaDataDir)
        sampleWeights = data.merge(prototypeWeights, how = 'left', left_on = 'prototype', right_on = prototypeWeights.index)['weight']
        data_encoded['weight'] = sampleWeights

        ############### Build dataset per prototype and concat #############

        # init
        # features_encoding = [col for col in data_encoded.columns.tolist() if col.startswith('prototype_')] # one-hot
        # features_encoding = ['targetMean'] # target mean
        features_encoding = ['targetMean', 'weight'] # target mean and weight

        ######### pre-allocate empty array
        trainX_sampleCount = 0
        trainY_sampleCount = 0
        valX_sampleCount = 0
        valY_sampleCount = 0

        for prototypeSelect in protoList:
            print()
            print('Get sample count for: ', prototypeSelect)
            data_3 = data_encoded.loc[data['prototype'] == prototypeSelect]
            protoClimate = [str(item[1]) for item in pairList_train if item[0] == prototypeSelect]
            trainX_t, trainY_t, valX_t, valY_t, _, _ = makeDatasets(protoClimate,
                                                                    data_3,
                                                                    lag,
                                                                    target,
                                                                    [],
                                                                    splitData_biRNN,
                                                                    allInTrain = True,
                                                                    )
            trainX_sampleCount += trainX_t.shape[0]
            trainY_sampleCount += trainY_t.shape[0]
            valX_sampleCount += valX_t.shape[0]
            valY_sampleCount += valY_t.shape[0]

        trainX = np.empty((trainX_sampleCount, len(lag) + 1, len(featureList + features_encoding)), dtype = np.float32)
        valX = np.empty((valX_sampleCount, len(lag) + 1, len(featureList + features_encoding)), dtype = np.float32)
        trainY = np.empty((trainY_sampleCount, 1, 1), dtype = np.float32)
        valY = np.empty((valY_sampleCount, 1, 1), dtype = np.float32)

        ####### make dataset
        trainX_sampleCount_accumulate = 0
        trainY_sampleCount_accumulate = 0
        valX_sampleCount_accumulate = 0
        valY_sampleCount_accumulate = 0

        for prototypeSelect in protoList:

            print()
            print('---------- Making datasets: ', prototypeSelect, ' ----------')

            data_2 = data_encoded.loc[data['prototype'] == prototypeSelect]

            # for saving memory
            data_encoded = data_encoded.loc[data['prototype'] != prototypeSelect]
            data = data.loc[data['prototype'] != prototypeSelect]

            protoClimate = [str(item[1]) for item in pairList_train if item[0] == prototypeSelect]
            trainX_0, trainY_0, valX_0, valY_0, _, _ = makeDatasets(protoClimate,
                                                            data_2,
                                                            lag,
                                                            target,
                                                            featureList + features_encoding,
                                                            splitData_biRNN,
                                                            allInTrain = True,
                                                            shuffle = True,
                                                            )

            trainX[trainX_sampleCount_accumulate: (trainX_sampleCount_accumulate + trainX_0.shape[0]), :, :] = trainX_0
            trainY[trainY_sampleCount_accumulate: (trainY_sampleCount_accumulate + trainY_0.shape[0]), :, :] = trainY_0
            valX[valX_sampleCount_accumulate: (valX_sampleCount_accumulate + valX_0.shape[0]), :, :] = valX_0
            valY[valY_sampleCount_accumulate: (valY_sampleCount_accumulate + valY_0.shape[0]), :, :] = valY_0

            trainX_sampleCount_accumulate += trainX_0.shape[0]
            trainY_sampleCount_accumulate += trainY_0.shape[0]
            valX_sampleCount_accumulate += valX_0.shape[0]
            valY_sampleCount_accumulate += valY_0.shape[0]

            del trainX_0, trainY_0, valX_0, valY_0

        sampleWeightTrain = trainX[:, 0, -1]
        sampleWeightVal = valX[:, 0, -1]
        trainX = trainX[:, :, :-1]
        valX = valX[:, :, :-1]

        np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/trainX.npy', trainX)
        np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/trainY.npy', trainY)
        np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/valX.npy', valX)
        np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/valY.npy', valY)
        # np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/sampleWeightTrain.npy', sampleWeightTrain)
        # np.save('./saved/allBuildingLevelData/randomSeed_' + str(randomSeed) + '/sampleWeightVal.npy', sampleWeightVal)


    ############### Model training #############
    tuner = kt.BayesianOptimization(
        biLSTM_tuner_global(),
        objective = 'val_loss',
        max_trials = tuneTrail,
        overwrite = True,
        directory = './tuner',
        project_name = 'biLSTM_global ' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    tuner.search(trainX, trainY,
                 epochs = maxEpoch,
                 callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3, min_delta=0, ), ],
                 validation_data = (valX, valY,
                                    # sampleWeightVal
                                    ),
                 # sample_weight = sampleWeightTrain,
                 )
    model = tuner.get_best_models(num_models=1)[0]
    model.build(input_shape = (None, trainX.shape[1], trainX.shape[2]))

    ########### predict ###########
    # for each of the prototype
    predictionDict = {}
    for prototypeSelect in protoList:

        # predict the test building-weather pairs whose prototype is in this loop
        protoClimate_predict = [item[1] for item in pairList_test if item[0] == prototypeSelect]
        if len(protoClimate_predict) == 0:
            print(prototypeSelect, ' is not included in test.')

        for weatherSelect in protoClimate_predict:
            print('    ---------- Building-Weather pair under estimation: ', prototypeSelect, '____', weatherSelect,
                  ' ----------')

            # get data of each weather
            weatherSelect = str(weatherSelect)
            data_energy = importRawData(
                dirEnergyTarget + '/' + prototypeSelect + '____' + weatherSelect + '.csv',
                col = target
                )
            data_weatherSelect = importWeatherData(dirWeatherTarget, weatherSelect)
            data_typical = importTypical(dirTypicalTarget, prototypeSelect,
                                         target, dayOfWeekJan1)

            # data_encodes = encodingDf[['prototype_' + prototypeSelect]].transpose() # one-hot encoding
            data_encodes = encodingDf[encodingDf['prototype'] == prototypeSelect][['targetMean']] # target-mean encoding
            data_encodes = pd.concat([data_encodes] * len(data_typical), ignore_index = True)

            data = pd.concat([data_energy, data_weatherSelect, data_typical, data_encodes], axis = 1)
            dataShort = data[[target] + featureList + data_encodes.columns.tolist()]
            sequences = sequencesGeneration(dataShort, lag, featureList + data_encodes.columns.tolist(), target)

            # estimation
            sequences_x = sequences[:, :, 1:]
            prediction = model.predict(sequences_x)

            # record prediction
            lagShift = (len(lag) + 1) // 2
            predictionDF = pd.DataFrame(prediction[:, 0], columns = ['estimate'])
            predictionDF['true'] = sequences[:, lagShift: (lagShift + 1), 0]
            predictionDF['DateTime'] = pd.date_range(start='2001-01-01 00:00:00', end='2001-12-31 23:00:00',
                                                     freq='H').to_series().iloc[lagShift: -lagShift].to_list()
            predictionDict[prototypeSelect + '____' + weatherSelect] = predictionDF

    return predictionDict