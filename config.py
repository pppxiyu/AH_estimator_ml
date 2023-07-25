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

modelName: 'naive', 'LSTM', 'biRNN', 'linear', 'mlp'
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
modelName = 'biLSTM'
tuneTrail = 1
maxEpoch = 500

# scaling up info
target_tractLevel = 'emission.surf'

# results saving for further eval
saveFolderHead = 'emissionSurf_biLSTM'
