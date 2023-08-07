import json
import time
from models.LSTM import *
from models.biLSTM import *
from models.biLSTM_global import *
from models.naive import *
from models.linear import *
from models.mlp import *
from utils.preprocess import *
from utils.tract import *
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

    # 0.0 randomness
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

    # 1.0 training and predicting
    pairListTrain, pairListTest = splitBuildingWeatherPair('./data/building_metadata/building_metadata.csv')
    with open(dirName + '/pairListTrain.json', 'w') as f:
        json.dump(pairListTrain, f)
    with open(dirName + '/pairListTest.json', 'w') as f:
        json.dump(pairListTest, f)

    if modelName == 'naive':
        prediction_tract = train_tract_naive(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                             pairListTrain, pairListTest,
                                             target = target_buildingLevel
                                             )
    if modelName == 'LSTM':
        prediction_tract = train_tract_LSTM(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                            pairListTrain, pairListTest,
                                            features,
                                            target = target_buildingLevel,
                                            lag = lag,
                                            tuneTrail=tuneTrail,
                                            maxEpoch=maxEpoch,
                                            )
    if modelName == 'biLSTM':
        prediction_tract = train_tract_biRNN(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                             pairListTrain, pairListTest,
                                             features,
                                             target = target_buildingLevel,
                                             lag = lag,
                                             tuneTrail = tuneTrail,
                                             maxEpoch = maxEpoch,
                                             )
    if modelName == 'linear':
        prediction_tract = train_tract_linear(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                             pairListTrain, pairListTest,
                                             features,
                                             target = target_buildingLevel,
                                             lag = lag,
                                             )
    if modelName == 'mlp':
        prediction_tract = train_tract_mlp(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                             pairListTrain, pairListTest,
                                             features,
                                             target = target_buildingLevel,
                                             lag = lag,
                                             )

    if modelName == 'biLSTM_global':
        prediction_tract = train_tract_biRNN_global(getAllPrototype('./data/hourly_heat_energy/sim_result_ann_WRF_2018_csv'),
                                                    pairListTrain, pairListTest,
                                                    features,
                                                    target = target_buildingLevel,
                                                    lag = lag,
                                                    tuneTrail = tuneTrail,
                                                    maxEpoch = maxEpoch,
                                                    metaDataDir = './data/building_metadata/building_metadata.csv',
                                                    randomSeed = randomSeed,
                                                    )

    if not os.path.exists(dirName + '/buildingLevel'):
        os.makedirs(dirName + '/buildingLevel')
    for key, df in prediction_tract.items():
        df.to_csv(dirName + '/buildingLevel/' + f'{key}.csv')

    # 1.1 norm the prediction
    # prototypeArea = getBuildingArea_prototype('C:/Users/xiyu/Downloads/output_data/output_data/EP_output/result_ann_WRF_2018',
    #                                              verbose = 0)
    with open('./data/building_metadata/prototypeArea.json', 'r') as f:
        prototypeArea = json.load(f)
    prediction_tract_norm = normalize_perM2(prediction_tract, pairListTest, prototypeArea)

    # 2.0 get metadata of tracts and remove tracts with weather not in the test pairs
    tractBuildingMeta = getBuildingArea_tracts('./data/building_metadata/building_metadata.csv', pairListTest)
    tractNameRemove = getTracts_remove('./data/building_metadata/building_metadata.csv', tractBuildingMeta)
    tractBuildingMeta = tractBuildingMeta[~tractBuildingMeta['id.tract'].isin(tractNameRemove)]

    # 2.1 scaling up
    estimate_tract = predict_tracts(prediction_tract_norm, tractBuildingMeta)

    endTime = time.time()
    executionTime = endTime - startTime

    # 3 get true data and remove tract with weather that is not in the test pairs
    true_tract = filterTractData(loadTractData('./data/hourly_heat_energy/annual_2018_tract.csv', target_tractLevel),
                                 estimate_tract)
    true_tract = true_tract[~true_tract.geoid.isin(tractNameRemove)]

    # 4 eval
    tractsDf = combineEstimateTrue(true_tract, estimate_tract, target_tractLevel)
    tractsDf.to_csv(dirName + '/' + 'tractsDF.csv')

    getTractLevelMetrics(tractsDf,
                         './saved/estimates_tracts/' + experimentLabel,
                         executionTime,
                         )
    plotPrototypeLevelMetrics(prediction_tract,
        './saved/estimates_tracts/' + experimentLabel,
        cv_mean_absolute_error_wAbs,
        'CVMAE_wAbs')
