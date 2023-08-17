import utils.tract as tr
import utils.eval as ev
import utils.vis as vis
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mode = 'singleExp_afterAnalysis'

    if mode == 'singleExp_resumeAnalysis':
        # 1.1 norm the prediction
        # prototypeArea = getBuildingArea_prototype('C:/Users/xiyu/Downloads/output_data/output_data/EP_output/result_ann_WRF_2018', verbose = 0)
        # used for generate the prototype areas, do not have to do it again
        experimentLabel = 'energyElec_biLSTM_2023-08-13-02-59-39'
        dir_trueTractData = './data/hourly_heat_energy/annual_2016_tract.csv'
        target_tractLevel = 'energy.elec'
        with open('./saved/estimates_tracts/' + experimentLabel + '/pairListTest.json', 'r') as f:
            pairListTest = json.load(f)

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

        # # output the metrics for each prototype
        # experimentLabel = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
        # predPrototypeLevel = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

        # # get metrics ar prototype
        # tr.metricPrototype(tr.metricPrototypeWeather(predPrototypeLevel,ev.cv_mean_absolute_error_wAbs),
        #                 'CVMAE_wAbs').to_csv('./saved/estimates_tracts/'+ experimentLabel + '/prototypeLevel' + 'CVMAE_wAbs' + '.csv',
        #                                      index = False)
        # vis.plotPrototypeLevelMetrics_plotly(predPrototypeLevel,
        #                                     './saved/estimates_tracts/' + experimentLabel,
        #                                     ev.cv_mean_absolute_error_wAbs,
        #                                     'CVMAE_wAbs')

        # # draw line chart using ground truth and estimation for a prototype
        # prototypeSelected = 'SingleFamily-2004____36'
        # prototypeDfSelected = predPrototypeLevel[prototypeSelected]
        # prototypeDfSelected['DateTime'] = prototypeDfSelected['DateTime'].apply(lambda x: x.replace('2001', '2018'))
        # prototypeDfSelected = prototypeDfSelected.iloc[700: 1440]
        # vis.plotPrototypeLevelLines_plotly(prototypeDfSelected,
        #                                    './paper/figs/' + 'line_' + prototypeSelected + '.html',
        #                                    prototypeSelected.split('_')[0] + '  ' + prototypeSelected.split('_')[-1]
        #                                    )

        # # draw line chart using ground truth and estimation for a microclimate zone
        # # get tract number
        # experimentLabel = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
        # selectedClimate = [65, 35, 109, 50]
        # vis.getTractIndexinWeatherZone(experimentLabel, selectedClimate, './data/building_metadata/building_metadata.csv')
        # # line charts
        # predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        # vis.plotTractLevelPredictionLine(predTractLevel, 6037600303, './paper/figs/line_tract6037600303.html', [3972,4331])

        # # draw census tract level cvmae with std shadow
        # experimentLabel = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
        # predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        # vis.plotTractLevelCVMAELine_wShadow(predTractLevel,
        #                                     # [4332, 5000], # July
        #                                     # [6540, 7283], # Oct
        #                                     # [8004, 8735], # Dec
        #                                     [2148, 2867], # April
        #                                     # [0, 8735], # all year
        #                                     './paper/figs/line_allTractCVMAEWithShadow_Apr.html')

        # # draw census tract level true value with std shadow
        # experimentLabel = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
        # predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        # vis.plotTractLevelTargetEstimate_wShadow(predTractLevel,
        #                                  'true',
        #                                  # [4332, 5000], # July
        #                                  # [6540, 7283], # Oct
        #                                  # [8004, 8735], # Dec
        #                                  # [2148, 2867], # April
        #                                  [0, 8735], # all year
        #                                  './paper/figs/line_allTractTargetWithShadow.html')

        # # print peak CVMAE and plot
        # experimentLabel = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
        # predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        # vis.plotPeaks_plotly(predTractLevel[predTractLevel.geoid == 6037101122], 'true', './paper/figs/peakValley_6037101122.html')
        # vis.plotDistPeakCVMAE(predTractLevel, './paper/figs/dist_peakCVMAE.html')

        # # output the metrics at tract level
        # predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
        # tr.getTractLevelMetrics(predTractLevel, './saved/estimates_tracts/' + experimentLabel)
        # vis.plotTractLevelPredictionLine(predTractLevel,
        #                              predTractLevel.geoid.sample(1).iloc[0],
        #                              './saved/estimates_tracts/' + experimentLabel)
        pass

    if mode == 'crossExp':
        ######### eval across experiments

        # prototype level metrics of several targets
        experimentLabels = [
            'energyElec_biLSTM_2023-08-14-03-11-39',
            # 'energyElec_biLSTM_2023-07-20-20-04-01',
        ]
        prototypeLevelMetrics = []
        for experiment in experimentLabels:
            prototypeLevelMetrics.append(
                tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experiment + '/buildingLevel'))
        vis.plotPrototypeLevelMetrics_plotly(prototypeLevelMetrics,
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
        experimentLabel = 'energyElec_biLSTM_2023-07-20-20-04-01'
        predPrototypeLevel = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')
        df = predPrototypeLevel['SingleFamily-2004____91']

        experimentLabel2 = 'energyElec_biLSTM_2023-08-14-03-11-39'
        predPrototypeLevel2 = tr.reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel2 + '/buildingLevel')
        df2 = predPrototypeLevel2['SingleFamily-2004____91']

        df['estimate'].iloc[2100:2300].plot()
        df['true'].iloc[2100:2300].plot()
        plt.legend()
        plt.show()

        df2['estimate'].iloc[2100:2300].plot()
        df2['true'].iloc[2100:2300].plot()
        plt.legend()
        plt.show()

        print()
