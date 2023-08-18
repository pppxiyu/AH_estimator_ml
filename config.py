import numpy as np

"""
BUILDING LEVEL WEATHER FEATURE:
'GLW', 'PSFC', Q2', 'RH', 'SWDOWN', 'T2', 'WINDD', 'WINDS'

BUILDING LEVEL TARGET:                                              TRACT LEVEL TARGET:
Environment:Site Total Surface Heat Emission to Air [J](Hourly)     emission.surf
Environment:Site Total Zone Exfiltration Heat Loss [J](Hourly)      emission.exfiltration
Environment:Site Total Zone Exhaust Air Heat Loss [J](Hourly)       emission.exhaust
SimHVAC:Air System Relief Air Total Heat Loss Energy [J](Hourly)    emission.ref
SimHVAC:HVAC System Total Heat Rejection Energy [J](Hourly)         emission.rej
Cooling:Electricity [J](Hourly)                                     
Electricity:Facility [J](Hourly)                                    energy.elec
NaturalGas:Facility [J](Hourly)                                     energy.gas

lag length must be a even number if biRNN is used.

modelName: 'naive', 'LSTM', 'biLSTM', 'linear', 'mlp'

dirTargetYear: a list of four elements. First element is the dir of energy data. Second is for weather data. Third is
    for typical target values. The last one is the tract level ground truth.
    Example for estimating 2016 whole year:
    [
    './data/hourly_heat_energy/sim_result_ann_WRF_2016_csv',
    './data/weather input/2016',
    './data/testrun',
    './data/hourly_heat_energy/annual_2016_tract.csv'
    ]
    Leave it as None if train and test are in the same year.

"""

# feature info
features = ['GLW', 
    'PSFC', 
    'Q2', 'RH', 'SWDOWN', 'T2', 'WINDD', 
    'WINDS',
    'Typical-Environment:Site Total Surface Heat Emission to Air [J](Hourly)',]
target_buildingLevel = 'Environment:Site Total Surface Heat Emission to Air [J](Hourly)'
lag = (
    (np.arange(24) + 1).tolist()
)

# model info
modelName = 'mlp'
tuneTrail = 1
maxEpoch = 500

testDataPer = 0.9

# scaling up info
target_tractLevel = 'emission.surf'

# results saving for further eval
saveFolderHead = 'emissionSurf_mlp_10PerData'

# random seed
randomSeed = 1

# target year
# dirTargetYear = [
#     './data/hourly_heat_energy/sim_result_ann_WRF_2016_csv',
#     './data/weather input/2016',
#     './data/testrun',
#     './data/hourly_heat_energy/annual_2016_tract.csv'
#     ]
dirTargetYear = None
dayOfWeekJan1 = 1


