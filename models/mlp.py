from utils.feature import *
from utils.preprocess import *
from datetime import datetime

from sklearn.metrics import mean_absolute_percentage_error

from sklearn.neural_network import MLPRegressor

def mlp_predict(sequenceList_test, model):
    predictionList = []
    for testX_0 in sequenceList_test:
        testX = testX_0[:, :-1, 1:]  # the time lag of energy/heat should not be included
        testX = testX.reshape(testX.shape[0], -1)
        prediction = model.predict(testX)
        predictionList.append(prediction)
    return predictionList

def train_tract_mlp(dirs, pairList_train, pairList_test, featureList, target, lag, dayOfWeekJan1):
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
    # for prototypeSelect in getAllPrototype(dirEnergy):
    for prototypeSelect in sorted(list(set([item[0] for item in pairList_test]))):

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
                                    1, # hard coded, because typical value is obtained in 2018
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

        # cancel the temporal dimension and de-stack the features of lagged timestamps
        trainX = np.concatenate((trainX, valX), axis = 0)
        trainY = np.concatenate((trainY, valY), axis = 0)
        trainX = trainX.reshape(trainX.shape[0], -1)
        trainY = trainY.reshape(trainY.shape[0], -1)
        print('Updated train_X shape is: ', trainX.shape)
        print('Updated train_Y shape is: ', trainY.shape)

        # train and save model
        model = MLPRegressor(
            hidden_layer_sizes = (100, 75, 50),
            early_stopping = True,
            n_iter_no_change = 10,
            validation_fraction = 0.15,
            learning_rate_init = 0.0005,
            )
        model.fit(trainX, trainY)

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
            data_typical = importTypical(dirTypicalTarget, prototypeSelect, target, dayOfWeekJan1) # for adding the typical
            data = pd.concat([data_energy, data_weatherSelect, data_typical], axis = 1)
            dataShort = data[[target] + featureList]
            sequences_weatherSelect = sequencesGeneration(dataShort, lag, featureList, target)

            # estimation
            sequences_weatherSelect_x = sequences_weatherSelect[:, :-1, 1:]
            sequences_weatherSelect_x = sequences_weatherSelect_x.reshape(sequences_weatherSelect_x.shape[0], -1)
            prediction = model.predict(sequences_weatherSelect_x)

            # record prediction
            predictionDF = pd.DataFrame(prediction, columns = ['estimate'])
            predictionDF['true'] = sequences_weatherSelect[:, -1, 0]

            print(mean_absolute_percentage_error(predictionDF.true, predictionDF.estimate))

            predictionDF['DateTime'] = pd.date_range(start = '2001-01-01 00:00:00',
                                                     end = '2001-12-31 23:00:00', freq = 'H').to_series().iloc[lag[-1]:].to_list()
            predictionDict[prototypeSelect + '____' + weatherSelect] = predictionDF

    return predictionDict
