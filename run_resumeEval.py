from utils.tract import *
import pandas as pd 


if __name__ == '__main__':

	experimentLabel = 'emissionExhaust_biLSTM_2023-07-23-22-33-50'
	predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
	predPrototypeLevel = reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

	getTractLevelMetrics(predTractLevel, './saved/estimates_tracts/' + experimentLabel)
	plotPrototypeLevelMetrics(predPrototypeLevel,
		'./saved/estimates_tracts/' + experimentLabel,
		cv_mean_absolute_error_wAbs,
		'CVMAE_wAbs')

	# plotTractLevelPredictionLine(predTractLevel,
	# 							 predTractLevel.geoid.sample(1).iloc[0],
	# 							 './saved/estimates_tracts/' + experimentLabel)
