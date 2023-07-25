# AH_estimator_ml

Deep learning-based estimation of AH and Energy load on 
the cencus tract level for a quick expasion of simulated data.

## Configuration
Edit the `config.py` to configure the training and estimation.

`features`: a list of string. Supports inputs including 
`'GLW'`, `'PSFC'`, `'Q2'`, `'RH'`, `'SWDOWN'`, `'T2'`, `'WINDD'`, `'WINDS'`, 
and `'Typical' + target_buildingLevel`.

`target_tractLevel`: string. Supports inputs in the `Short Name`
column of the following table.

`target_buildingLevel`: string. Supports inputs in the `Full Name`
column of the following table.



| Short Name | Full Name                                                         |
|------------|-------------------------------------------------------------------|
| emission.surf  | Environment:Site Total Surface Heat Emission to Air \[J](Hourly)  |
| emission.exfiltration | Environment:Site Total Zone Exfiltration Heat Loss \[J](Hourly)   | 
| emission.exhaust | Environment:Site Total Zone Exhaust Air Heat Loss \[J](Hourly)    |
| emission.ref | SimHVAC:Air System Relief Air Total Heat Loss Energy \[J](Hourly) |
| emission.rej | SimHVAC:HVAC System Total Heat Rejection Energy \[J](Hourly)      |
| energy.elec | Electricity:Facility \[J](Hourly)                                 |
| energy.gas | NaturalGas:Facility \[J](Hourly)                                  |


`modelName`: string. Supports `'naive'`, `'LSTM'`, `'biRNN'`, `'linear'`, 
`'mlp'`.

`tuneTrail`: int. Number of trails in hyper-parameter tuning. Only works
for `'LSTM'` and `'biLSTM'`.

`maxEpoch`: int. Max epoch count for `'LSTM'` and `'biLSTM'` training.

`lag`: list of int. The index of time lags in each sequence, ranging from 1. 
For example, in the case of using 4 time lags to estimate 1 timestamp forward with `'LSTM'`, 
`lag` should be `[1, 2, 3, 4]`. In the case of using 2 timestamps both in the past
and future to estimate the timestamp in the middle with `'biLSTM'`, also use
`[1, 2, 3, 4]`. Please note if `'biLSTM'` is used, the length of `lag` must be 
an even number.

`saveFolderHead`: string. Recommend name it using the `target_tractLevel`
and the `modelName`. For example, `energyElec_biLSTM` stands for the experiment
using `'biLSTM'` for estimating `'energy.elec'`.

## Datasets
Download the dataset from `  ` and name it as `data` in the root dir of the project.
Please refer to the work by [Xu et. al.](https://github.com/IMMM-SFA/xu_etal_2022_sdata)
for the explaination of the datasets.

## Installation and running

Install project environment:
```
pip install -r requirements.txt
```
Training and estimation:
```
python run.py
```

## Reference
