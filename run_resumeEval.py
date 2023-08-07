from utils.tract import *
import pandas as pd 


if __name__ == '__main__':

	experimentLabel = 'energyElec_biLSTM_global_2023-08-03-00-49-39'
	# predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
	predPrototypeLevel = reloadPrototypeLevelPred('./saved/estimates_tracts/' + experimentLabel + '/buildingLevel')

	# getTractLevelMetrics(predTractLevel, './saved/estimates_tracts/' + experimentLabel)
	plotPrototypeLevelMetrics(predPrototypeLevel,
		'./saved/estimates_tracts/' + experimentLabel,
		mean_squared_error,
		'MSE')

	# plotTractLevelPredictionLine(predTractLevel,
	# 							 predTractLevel.geoid.sample(1).iloc[0],
	# 							 './saved/estimates_tracts/' + experimentLabel)
