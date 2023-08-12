# AH_estimator_ml

## Abstract

## I/O
### Inputs
Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1RWX9ef1bM4drVp5AVWS11xkaZFG_q-DO?usp=drive_link) 
and name it as `data` in the root dir of the project.
Please refer to the work by [Xu et. al.](https://github.com/IMMM-SFA/xu_etal_2022_sdata)
for the explaination of the datasets.

Create a dir `./saved/estimates_tracts` in the root dir of the project.

### Outputs
After configure and run the program, a folder that contains all the outputs for
this run will be generated under `./saved/estimates_tracts`. The output folder
is named with `target_model_experimentTime`. For example, `energyElec_biLSTM_2023-07-21-21-39-29`
is the output folder for estimating electricity with bi-directional LSTM at
21:39:29 07/21 2023 (available targets and model names are introduced below).

`pairListTest.json` and `pairListTrain.json` contains the `prototype-weather` pairs
used for test and training.

The `buildingLevel` folder under the `./saved/estimates_tracts/target_model_experimentTime` contains
the intermediate result. Each `.csv` file contains the hourly estimation of the
target at the building prototype level.

`tractsDF.csv` is the estimation of the target at the census tract level.

`config.py` shows the configuration of this experiment, and other files in the 
`./saved/estimates_tracts/target_model_experimentTime` folder are used for 
the evaluation and visualization of the estimations.




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
`'mlp'`, and `biRNN_global`.

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

`randomSeed`: int. The random seed used by numpy.random.

`dirTargetYear`: `None` or list. In default, `dirTargetYear` is set as `None`. In 
this case, the microclimates zones in the 2018 is split into training and testing 
set. However, in real use case, 2018 microclimate zones are used for training. The trained
model will estimate the targets in another year. `dirTargetYear` should be set as a 
list, indicating the dir of input features and ground truth (if any) for test.
The first element is the dir of energy data. The second is for weather data. The third is 
for typical target values. The last one is the tract level ground truth. Example for estimating 2016 whole year:
```
[
'./data/hourly_heat_energy/sim_result_ann_WRF_2016_csv',
'./data/weather input/2016',
'./data/testrun',
'./data/hourly_heat_energy/annual_2016_tract.csv'
]
```


## Global estimation option

Global estimation means using one single model to do the estimation for all prototypes. It is expected to 
generate better accuracy in some cases, because different prototypes share part of the data generation process,
and it works as a simple multitasks learning architecture. However, it should be noted that the number of
samples for the model will increase dramatically, as the samples for each separate model of each prototype
has been stacked togather. The total size of training and validation `numpy` array with `float32` type
is about 25GB. Servers with 30GB GPU and CPU memory are recommended to be used for the global esitmation option.




## Installation and running

Install project environment:
```
pip install -r requirements.txt
```
Training and estimation:
```
python run.py
```

## Auxiliary files
`run_preAnalysis.py` conducts preliminary analysis (e.g., visualization, autocorelation plot) 
on the raw data.

`run_prototypeLevel.py` runs the model for the selected prototype. Keep it for
debugging purposes.

`run_resumeEval.py` and `run_resumeScaleUp.py` are kept for debugging purposes. They 
reload the saved estimations on the prototype level and redo the scaling up
to census tract level or the metrics calculation/visualiztion. Please also
use `run_resumeEval.py` to run new evaluation functions, 
if further custom evaluation of the estimation is needed
