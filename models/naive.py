import warnings
from utils.preprocess import *
from utils.tract import *

def train_tract_naive(protoList, pairList_train, pairList_test, target):
    # USE:
    # INPUT: all prototype list, pairs for train, pairs for test, featrue names, target name, lag list, ifTune True or False
    # OUTPUT: dict, each value is the prediction for a pair in the test pair set

    # for each of the prototype
    predictionDict = {}
    for prototypeSelect in protoList:

        ########### train ###########
        print()
        print('---------- Modeling: ', prototypeSelect, ' ----------')

        # get weathers names in train_pairs for the prototype
        protoClimate = [str(item[1]) for item in pairList_train if item[0] == prototypeSelect]
        if len(protoClimate) < 1:
            warnings.warn("Some building type is missing in training dataset.")

        # get weather data in train_pairs for the prototype
        data = getAllData4Prototype(prototypeSelect, protoClimate,
                                    './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv',
                                    './data/weather input',
                                    './data/testrun',
                                    target
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
                './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv/' + prototypeSelect + '____' + weatherSelect + '.csv',
                col = target,
            )

            # record prediction
            predictionDF = pd.DataFrame(prediction, columns=['estimate'])
            predictionDF['true'] = data_energy[target]
            predictionDF['DateTime'] = pd.date_range(start='2018-01-01 00:00:00', end='2018-12-31 23:00:00',
                                                     freq='H').to_series().to_list()
            predictionDict[prototypeSelect + '____' + weatherSelect] = predictionDF

    return predictionDict
