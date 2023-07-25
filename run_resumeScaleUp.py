import json
from utils.tract import *
import config


if __name__ == '__main__':

    experimentLabel = 'energyElec_biLSTM_2023-07-21-21-39-29'

    target_tractLevel = config.target_tractLevel

    # # 1.1.1 load saved results
    prediction_tract = reloadPrototypeLevelPred(
        './saved/estimates_tracts/' + experimentLabel + '/buildingLevel')
    with open('./saved/estimates_tracts/' + experimentLabel + '/pairListTest.json', 'r') as f:
        pairListTest = json.load(f)
    with open('./saved/estimates_tracts/' + experimentLabel + '/pairListTrain.json', 'r') as f:
        pairListTrain = json.load(f)

    # 1.2 norm the prediction
    # prototypeArea = getBuildingArea_prototype('C:/Users/xiyu/Downloads/output_data/output_data/EP_output/result_ann_WRF_2018',
    #                                              verbose = 0)
    with open('./data/building_metadata/prototypeArea.json', 'r') as f:
        prototypeArea = json.load(f)
    prediction_tract_norm = normalize_perM2(prediction_tract, pairListTest, prototypeArea)

    # 2.1 get metadata of tracts and remove tracts with weather not in the test pairs
    tractBuildingMeta = getBuildingArea_tracts('./data/building_metadata/building_metadata.csv', pairListTest)
    tractNameRemove = getTracts_remove('./data/building_metadata/building_metadata.csv', tractBuildingMeta)
    tractBuildingMeta = tractBuildingMeta[~tractBuildingMeta['id.tract'].isin(tractNameRemove)]

    # 2.2 scaling up
    estimate_tract = predict_tracts(prediction_tract_norm, tractBuildingMeta)

    # 3 get true data and remove tract with weather that is not in the test pairs
    true_tract = filterTractData(loadTractData('./data/hourly_heat_energy/annual_2018_tract.csv', target_tractLevel),
                                 estimate_tract)
    true_tract = true_tract[~true_tract.geoid.isin(tractNameRemove)]

    # 4 eval
    tractsDf = combineEstimateTrue(true_tract, estimate_tract, target_tractLevel)
    tractsDf.to_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
    