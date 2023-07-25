from utils.tract import *
import pandas as pd 


if __name__ == '__main__':

	experimentLabel = 'energyElec_biLSTM_2023-07-21-21-39-29'
	predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
	predPrototypeLevel = reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

	# getTractLevelMetrics(predTractLevel, './saved/estimates_tracts/' + experimentLabel)
	# plotPrototypeLevelMetrics(predPrototypeLevel,
	# 	'./saved/estimates_tracts/' + experimentLabel,
	# 	cv_mean_absolute_error,
	# 	'CVMAE')

	plotTractLevelPredictionLine(predTractLevel,
								 predTractLevel.geoid.sample(1).iloc[0],
								 './saved/estimates_tracts/' + experimentLabel)
