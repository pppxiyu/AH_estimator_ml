import warnings
from utils.preprocess import *
from utils.tract import *

def train_tract_naive(dirs, pairList_train, pairList_test, target, dayOfWeekJan1):
    # USE:
    # INPUT: all prototype list, pairs for train, pairs for test, featrue names, target name, lag list, ifTune True or False
    # OUTPUT: dict, each value is the prediction for a pair in the test pair set

    dirEnergy = dirs[0]
    dirWeather = dirs[1]
    dirTypical = dirs[2]
    try:
        dirEnergyTarget = dirs[3]

    except:
        dirEnergyTarget = dirEnergy
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
                                    1, # hard coded, because typical value is obtained in 2018
                                    )
        data = data[['Date/Time', target]]

        # calculate  the average among all training weathers
        dataGrouped = data.groupby('Date/Time').mean()

        ########### predict ###########

        # copy the average as prediction
        prediction = dataGrouped[target].values

        # get the test building-weather pairs whose prototype is in this loop
        protoClimate_predict = [item[1] for item in pairList_test if item[0] == prototypeSelect]
        if len(protoClimate_predict) == 0:
            print('A model trained is not used in test.')

        for weatherSelect in protoClimate_predict:
            print('    ---------- Building-Weather pair under estimation: ', prototypeSelect, '____', weatherSelect,
                  ' ----------')
            # get true data
            weatherSelect = str(weatherSelect)
            data_energy = importRawData(
                dirEnergyTarget + '/' + prototypeSelect + '____' + weatherSelect + '.csv',
                col = target,
            )

            # record prediction
            predictionDF = pd.DataFrame(prediction, columns=['estimate'])
            predictionDF['true'] = data_energy[target]
            predictionDF['DateTime'] = pd.date_range(start='2001-01-01 00:00:00', end='2001-12-31 23:00:00',
                                                     freq='H').to_series().to_list()
            predictionDict[prototypeSelect + '____' + weatherSelect] = predictionDF

    return predictionDict
