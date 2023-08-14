import utils.tract as tr
import utils.eval as ev
import pandas as pd
import json
import os

if __name__ == '__main__':

    mode = 'others'

    if mode == 'singleExp_resumeAnalysis':
        # 1.1 norm the prediction
        # prototypeArea = getBuildingArea_prototype('C:/Users/xiyu/Downloads/output_data/output_data/EP_output/result_ann_WRF_2018', verbose = 0)
        # used for generate the prototype areas, do not have to do it again
        experimentLabel = 'energyElec_biLSTM_2023-08-13-02-59-39'
        dir_trueTractData = './data/hourly_heat_energy/annual_2016_tract.csv'
        target_tractLevel = 'energy.elec'
        with open('./saved/estimates_tracts/' + experimentLabel + '/pairListTest.json', 'r') as f:
            pairListTest = json.load(f)
        #
        # csvPath = './saved/estimates_tracts/' + experimentLabel + '/buildingLevel'
        # csv_files = [f for f in os.listdir(csvPath) if f.endswith('.csv')]
        # prediction_tract = {}
        # for file in csv_files:
        #     file_path = os.path.join(csvPath, file)
        #     df = pd.read_csv(file_path)
        #     prediction_tract[file[:-4]] = df

        dir_buildingMeta = './data/building_metadata/building_metadata.csv'
        predPrototypeLevel = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

        with open('./data/building_metadata/prototypeArea.json', 'r') as f:
            prototypeArea = json.load(f)
        prediction_tract_norm = tr.normalize_perM2(predPrototypeLevel, pairListTest, prototypeArea)


        # 2.0 get metadata of tracts and remove tracts with the weather not in the test pairs
        tractBuildingMeta = tr.getBuildingArea_tracts(dir_buildingMeta, pairListTest)

        tractNameRemove = tr.getTracts_remove(dir_buildingMeta, tractBuildingMeta)
        tractBuildingMeta = tractBuildingMeta[~tractBuildingMeta['id.tract'].isin(tractNameRemove)]

        # 2.1 scaling up
        estimate_tract = tr.predict_tracts(prediction_tract_norm, tractBuildingMeta)


        # 3 get true data and remove tracts with weather that is not in the test pairs
        true_tract = tr.filterTractData(tr.loadTractData(dir_trueTractData, target_tractLevel),
                                        estimate_tract)
        true_tract = true_tract[~true_tract.geoid.isin(tractNameRemove)]

        # 4 eval
        tractsDf = tr.combineEstimateTrue(true_tract, estimate_tract, target_tractLevel)
        tractsDf.to_csv('./saved/estimates_tracts/' + experimentLabel + '/' + 'tractsDF.csv')

        tr.getTractLevelMetrics(tractsDf,
                                './saved/estimates_tracts/' + experimentLabel,
                                0.1,
                                )
        tr.plotPrototypeLevelMetrics(predPrototypeLevel,
                                     './saved/estimates_tracts/' + experimentLabel,
                                     ev.cv_mean_absolute_error_wAbs,
                                     'CVMAE_wAbs')


    if mode == 'singleExp_afterAnalysis':
        ######## eval for each experiment
        experimentLabel = 'energyElec_biLSTM_2023-07-20-20-04-01'
        predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        predPrototypeLevel = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

        # output the metrics for each prototype
        tr.metricPrototype(tr.metricPrototypeWeather(predPrototypeLevel,ev.cv_mean_absolute_error_wAbs),
                        'CVMAE_wAbs').to_csv('./saved/estimates_tracts/'+ experimentLabel + '/prototypeLevel' + 'CVMAE_wAbs' + '.csv',
                                             index = False)
        tr.plotPrototypeLevelMetrics_plotly(predPrototypeLevel,
                                            './saved/estimates_tracts/' + experimentLabel,
                                            ev.cv_mean_absolute_error_wAbs,
                                            'CVMAE_wAbs')

        # output the metrics at tract level
        tr.getTractLevelMetrics(predTractLevel, './saved/estimates_tracts/' + experimentLabel)
        tr.plotTractLevelPredictionLine(predTractLevel,
                                     predTractLevel.geoid.sample(1).iloc[0],
                                     './saved/estimates_tracts/' + experimentLabel)

    if mode == 'crossExp':
        ######### eval across experiments

        # prototype level metrics of several targets
        experimentLabels = [
            'energyElec_biLSTM_2023-07-20-20-04-01',
            # 'energyElec_biLSTM_2023-07-20-20-04-01',
        ]
        prototypeLevelMetrics = []
        for experiment in experimentLabels:
            prototypeLevelMetrics.append(
                tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experiment + '/buildingLevel'))
        tr.plotPrototypeLevelMetrics_plotly(prototypeLevelMetrics,
                                            [
                                                'nMAE_ELEC',
                                                # 'nMAE_ELEC'
                                            ],
                                            [
                                                'rgb(55, 83, 109)',
                                                # 'rgb(26, 118, 255)',
                                            ],
                                            ev.cv_mean_absolute_error_wAbs,
                                            './paper/figs',
                                            )

    if mode == 'others':
        ######## other test
        experimentLabel = 'energyElec_biLSTM_2023-08-13-14-01-13'
        predPrototypeLevel = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')
        df = list(predPrototypeLevel.values())[0]
        print()
