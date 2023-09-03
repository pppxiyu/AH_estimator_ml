import json
import time
from models.LSTM import *
from models.biLSTM import *
from models.biLSTM_global import *
from models.naive import *
from models.linear import *
from models.mlp import *

# from utils.tract import *
import utils.preprocess as pp
import utils.tract as tr

from utils.eval import cv_mean_absolute_error
import config

if __name__ == '__main__':

    startTime = time.time()

    # 0.0 config
    features = config.features
    target_buildingLevel = config.target_buildingLevel
    lag = config.lag
    modelName = config.modelName
    tuneTrail = config.tuneTrail
    maxEpoch = config.maxEpoch
    target_tractLevel = config.target_tractLevel
    saveFolderHead = config.saveFolderHead
    randomSeed = config.randomSeed
    dirTargetYear = config.dirTargetYear
    dayOfWeekJan1 = config.dayOfWeekJan1
    testDataPer = config.testDataPer

    dir_buildingMeta = './data/building_metadata/building_metadata.csv'
    dir_energyData = './data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'
    dir_weatherData = './data/weather input/2018'
    dir_typicalData = './data/testrun'
    dirList = [dir_energyData, dir_weatherData, dir_typicalData]
    dir_trueTractData = './data/hourly_heat_energy/annual_2018_tract.csv'

    if dirTargetYear != None:
        dirList.append(dirTargetYear[0])
        dirList.append(dirTargetYear[1])
        dirList.append(dirTargetYear[2])
        dir_trueTractData = dirTargetYear[3]

    np.random.seed(randomSeed)

    # 0.1 create a dir for saving results
    currentTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experimentLabel = saveFolderHead + '_' + str(currentTime)
    dirName = './saved/estimates_tracts/' + saveFolderHead + '_' + str(currentTime)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    # 0.2 save config file
    with open("./config.py", 'rb') as src_file:
        with open(dirName + '/' + "./config.py", 'wb') as dst_file:
            dst_file.write(src_file.read())


    # 1.0 set training and testing set
    # Option 1: split data using microclimates
    # pairListTrain, pairListTest = tr.splitBuildingWeatherPair_byWeather(dir_buildingMeta, 0.85)
    # if dirTargetYear != None:
    #     pairListTrain = pairListTrain + pairListTest
    #     pairListTest = pairListTrain

    # Option 2: split data using microclimates for each prototype
    pairListTrain, pairListTest = tr.splitBuildingWeatherPair_byProto(dir_buildingMeta, testDataPer)

    # Option 3: only for debugging, it can limit the train and test to selected prototype-weather pairs
    # pairListTrain = pairListTrain[0:1]
    # pairListTest = pairListTest[0:1]
    # pairListTrain = [i for i in pairListTrain if i[0] == 'SingleFamily-2004']
    # pairListTest = [i for i in pairListTest if i[0] == 'SingleFamily-2004']

    with open(dirName + '/pairListTrain.json', 'w') as f:
        json.dump(pairListTrain, f)
    with open(dirName + '/pairListTest.json', 'w') as f:
        json.dump(pairListTest, f)

    # 1.1 train
    if modelName == 'naive':
        prediction_tract = train_tract_naive(dirList,
                                             pairListTrain, pairListTest,
                                             target = target_buildingLevel,
                                             dayOfWeekJan1 =dayOfWeekJan1,
                                             )
    if modelName == 'LSTM':
        prediction_tract = train_tract_LSTM(dirList,
                                            pairListTrain, pairListTest,
                                            features,
                                            target = target_buildingLevel,
                                            lag = lag,
                                            tuneTrail=tuneTrail,
                                            maxEpoch=maxEpoch,
                                            dayOfWeekJan1 = dayOfWeekJan1
                                            )
    if modelName == 'biLSTM':
        prediction_tract = train_tract_biRNN(dirList,
                                             pairListTrain, pairListTest,
                                             features,
                                             target = target_buildingLevel,
                                             lag = lag,
                                             tuneTrail = tuneTrail,
                                             maxEpoch = maxEpoch,
                                             dayOfWeekJan1 = dayOfWeekJan1,
                                             )
    if modelName == 'linear':
        prediction_tract = train_tract_linear(dirList,
                                             pairListTrain, pairListTest,
                                             features,
                                             target = target_buildingLevel,
                                             lag = lag,
                                              dayOfWeekJan1 = dayOfWeekJan1,
                                             )
    if modelName == 'mlp':
        prediction_tract = train_tract_mlp(dirList,
                                           pairListTrain, pairListTest,
                                           features,
                                           target = target_buildingLevel,
                                           lag = lag,
                                           dayOfWeekJan1 = dayOfWeekJan1,
                                           )

    if modelName == 'biLSTM_global':
        prediction_tract = train_tract_biRNN_global(pp.getAllPrototype(dir_energyData),
                                                    pairListTrain, pairListTest,
                                                    features,
                                                    target = target_buildingLevel,
                                                    lag = lag,
                                                    tuneTrail = tuneTrail,
                                                    maxEpoch = maxEpoch,
                                                    metaDataDir = dir_buildingMeta,
                                                    randomSeed = randomSeed,
                                                    dayOfWeekJan1 = dayOfWeekJan1,
                                                    )

    if not os.path.exists(dirName + '/buildingLevel'):
        os.makedirs(dirName + '/buildingLevel')
    for key, df in prediction_tract.items():
        df.to_csv(dirName + '/buildingLevel/' + f'{key}.csv')

    # 1.2 norm the prediction
    # prototypeArea = getBuildingArea_prototype('C:/Users/xiyu/Downloads/output_data/output_data/EP_output/result_ann_WRF_2018', verbose = 0)
    # used for generate the prototype areas, do not have to do it again

    with open('./data/building_metadata/prototypeArea.json', 'r') as f:
        prototypeArea = json.load(f)
    prediction_tract_norm = tr.normalize_perM2(prediction_tract, pairListTest, prototypeArea)


    # 2.0 get metadata of tracts and remove tracts with the weather not in the test pairs
    tractBuildingMeta = tr.getBuildingArea_tracts(dir_buildingMeta, pairListTest)
    # tractNameRemove = tr.getTracts_remove(dir_buildingMeta, tractBuildingMeta)
    # tractBuildingMeta = tractBuildingMeta[~tractBuildingMeta['id.tract'].isin(tractNameRemove)]

    # 2.1 scaling up
    estimate_tract = tr.predict_tracts(prediction_tract_norm, tractBuildingMeta)

    endTime = time.time()
    executionTime = endTime - startTime


    # 3 get true data and remove tracts with weather that is not in the test pairs
    true_tract = tr.filterTractData(tr.loadTractData(dir_trueTractData, target_tractLevel),
                                    estimate_tract)
    # true_tract = true_tract[~true_tract.geoid.isin(tractNameRemove)]

    # 4 eval
    tractsDf = tr.combineEstimateTrue(true_tract, estimate_tract, target_tractLevel)
    tractsDf.to_csv(dirName + '/' + 'tractsDF.csv')

    tr.getTractLevelMetrics(tractsDf,
                            './saved/estimates_tracts/' + experimentLabel,
                            executionTime,
                            )
    tr.plotPrototypeLevelMetrics(prediction_tract,
                                 './saved/estimates_tracts/' + experimentLabel,
                                 cv_mean_absolute_error_wAbs,
                                 'CVMAE_wAbs')
