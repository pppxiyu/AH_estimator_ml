import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from utils.feature import *
from utils.preprocess import *
from datetime import datetime

# LSTM
def LSTM_train(lr, batch_size, unit, trainX, trainY, valX, valY):
    model_LSTM = tf.keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(unit),
        keras.layers.Dense(1),
    ])
    model_LSTM.compile(optimizer = keras.optimizers.Adam(learning_rate = lr), loss = "mse")
    model_LSTM.fit(
        x = trainX,
        y = trainY,
        batch_size = batch_size,
        epochs = 500,
        validation_data = (valX, valY),
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss",
#                 min_delta = 0.0001,
                patience = 3),
        ],
        verbose = 1,
    )
    return model_LSTM

def LSTM_train_predict(lr, batch_size, unit, trainX, trainY, valX, valY, sequenceList_test):
    model_LSTM = tf.keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(unit),
        keras.layers.Dense(1),
    ])
    model_LSTM.compile(optimizer = keras.optimizers.Adam(learning_rate = lr), loss = "mse")
    model_LSTM.fit(
        x = trainX,
        y = trainY,
        batch_size = batch_size,
        epochs = 500,
        validation_data = (valX, valY),
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss",
#                 min_delta = 0.0001,
                patience = 3),
        ],
        verbose = 1,
    )

    predictionList = []
    for testX_0 in sequenceList_test:
        testX = testX_0[:, :-1, 1:]  # the time lag of energy/heat should not be included
        prediction = model_LSTM.predict(testX)
        predictionList.append(prediction)
    return predictionList

class LSTM_tuner(kt.HyperModel):
    def build(self, hp):
        hp_units = hp.Int('units', min_value = 256, max_value = 512, step = 128)
        model = tf.keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(hp_units),
            keras.layers.Dense(1),
#             keras.layers.Dense(2), # two timestamp forward
        ])
        hp_lr = hp.Float('learningRate', 0.0005, 0.0015, step = 0.0005)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_lr), loss = "mse")
        return model

    def fit(self, hp, model, *args, **kwargs):
        hp_batchSize = hp.Choice('batchSize', values = [128, 512, 1024])
        return model.fit(
            *args,
            batch_size = hp_batchSize,
            **kwargs,
        )

def LSTM_predict(sequenceList_test, model = None, tuner = None):
    if model is not None and tuner is not None:
        print('Only one of model and tuner should be input.')
        return

    if tuner:
        model = tuner.get_best_models(num_models = 1)[0]
        inputShape = sequenceList_test[0][:, :-1, 1:].shape
        model.build(input_shape = (None, inputShape[1], inputShape[2]))

    predictionList = []
    for testX_0 in sequenceList_test:
        testX = testX_0[:, :-1, 1:]  # the time lag of energy/heat should not be included
#         testX = testX_0[:, :-2, 1:]  # 2 timestamp forward
        prediction = model.predict(testX)
        predictionList.append(prediction)
    return predictionList

def train_tract_LSTM(dirs, pairList_train, pairList_test, featureList, target, lag, tuneTrail, maxEpoch):
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

    # for each of the prototype
    predictionDict = {}
    for prototypeSelect in getAllPrototype(dirEnergy):

        ########### train ###########
        print()
        print('---------- Modeling: ', prototypeSelect, ' ----------')

        # get weathers names in train_pairs for the prototype
        protoClimate = [str(item[1]) for item in pairList_train if item[0] == prototypeSelect]
        if len(protoClimate) < 1:
            warnings.warn("Some building type is missing in training dataset.")

        # get weather data in train_pairs for the prototype
        data = getAllData4Prototype(prototypeSelect, protoClimate,
                                    dirEnergy,
                                    dirWeather,
                                    dirTypical,
                                    target,
                                    )
        # build datasets
        trainX, trainY, valX, valY, _, _ = makeDatasets(protoClimate,
                                                        data,
                                                        lag,
                                                        target,
                                                        featureList,
                                                        splitData,
                                                        allInTrain = True,
                                                       )
        # train and save model
        tuner = kt.BayesianOptimization(
            LSTM_tuner(),
            objective = "val_loss",
            max_trials = tuneTrail,
            overwrite = True,
            directory = "./tuner",
            project_name = 'LSTM ' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        tuner.search(trainX, trainY,
                           epochs = maxEpoch,
                           callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience = 3, min_delta = 0,),],
                           validation_data = (valX, valY),
                )
        model = tuner.get_best_models(num_models = 1)[0]
        model.build(input_shape = (None, len(lag), trainX.shape[2]))

        ########### predict ###########

        # predict the test building-weather pairs whose prototype is in this loop
        protoClimate_predict = [item[1] for item in pairList_test if item[0] == prototypeSelect]
        if len(protoClimate_predict) == 0:
            print('A model trained is not used in test.')

        for weatherSelect in protoClimate_predict:
            print('    ---------- Building-Weather pair under estimation: ', prototypeSelect, '____', weatherSelect, ' ----------')

            # get data of each weather
            weatherSelect = str(weatherSelect)
            data_energy = importRawData(dirEnergyTarget + '/' + prototypeSelect + '____' + weatherSelect + '.csv',
                                        col = target
                                        )
            data_weatherSelect = importWeatherData(dirWeatherTarget, weatherSelect)
            data_typical = importTypical(dirTypicalTarget, prototypeSelect, target) # for adding the typical
            data = pd.concat([data_energy, data_weatherSelect, data_typical], axis = 1)
            dataShort = data[[target] + featureList]
            sequences_weatherSelect = sequencesGeneration(dataShort, lag, featureList, target)

            # estimation
            sequences_weatherSelect_x = sequences_weatherSelect[:, :-1, 1:]
            prediction = model.predict(sequences_weatherSelect_x)

            # record prediction
            predictionDF = pd.DataFrame(prediction[:, 0], columns = ['estimate'])
            predictionDF['true'] = sequences_weatherSelect[:, -1, 0]
            predictionDF['DateTime'] = pd.date_range(start = '2018-01-01 00:00:00',
                                                     end = '2018-12-31 23:00:00', freq = 'H').to_series().iloc[lag[-1]:].to_list()
            predictionDict[prototypeSelect + '____' + weatherSelect] = predictionDF

    return predictionDict