import pandas as pd
import os

def parseDatetime24(row):
    dt = row['Date/Time']
    if '24:00:00' in dt:
        dt = dt.replace(' 24:00:00', '23:00:00')
    else:
        dt = (pd.to_datetime(dt, format = '%Y/%m/%d  %H:%M:%S') - pd.Timedelta(hours = 1)).strftime('%Y/%m/%d %H:%M:%S')
    row['Date/Time'] = dt
    return row

def importRawData(addr, col):
    data = pd.read_csv(addr,
                       usecols = ['Date/Time',
                                  col,
                                 ])
    data[col] = data[col] / 3.6e+6
    data['Date/Time'] = pd.date_range('2018-01-01 00:00:00', '2018-12-31 23:00:00', freq = 'H')

#     data['Date/Time'] = data.apply(lambda row: row['Date/Time'].lstrip(), axis = 1)
#     data['Date/Time'] = "2018/" + data['Date/Time']
#     data['Date/Time'] = data[['Date/Time']].apply(parseDatetime24, axis = 1)['Date/Time']
#     data['Date/Time'] = pd.to_datetime(data['Date/Time'], format = '%Y/%m/%d  %H:%M:%S')
    return data

def importWeatherData(directory, climateID, length = 8760):
    # USE: import weather data for given climate ID
    # INPUT: dir, the dir of weather data 
    directory = directory + '/'
    df = pd.DataFrame({'Climate': [climateID] * length})
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                weather = pd.read_csv(file)
                df[filename.replace('.csv', '')] = weather[climateID]
    return df

def importTypical(addr, prototypeName, selectCol):
    data = pd.read_csv(addr + '/' + prototypeName + '/eplusout.csv', usecols = ['Date/Time', selectCol])
    data = data[[selectCol]]
#     data['Date/Time'] = pd.date_range('2018-01-01 00:00:00', '2018-12-31 23:00:00', freq = 'H')
#     data['Date/Time'] = data.apply(lambda row: row['Date/Time'].lstrip(), axis = 1)
#     data['Date/Time'] = "2018/" + data['Date/Time']
#     data['Date/Time'] = data[['Date/Time']].apply(parseDatetime24, axis = 1)['Date/Time']
#     data['Date/Time'] = pd.to_datetime(data['Date/Time'], format = '%Y/%m/%d  %H:%M:%S')
    
    data[selectCol] = data[selectCol] / 3.6e+6
    data = data.rename(columns = {selectCol: 'Typical-' + selectCol})
    
    return data

def getAllPrototype(directory):
    # USE: get all prototype
    # INPUT: the dir of sim results
    prototypeList = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            prototypeList.append(filename.split('____')[0])
    prototypeList = list(set(prototypeList))
    return sorted(prototypeList, key = str.lower)

def getAllClimates(directory):
    # USE: get the list of climate zone names
    # INPUT: dir, dir of energy data 
    
    climateList = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            climateList.append(filename.split('____')[1].replace('.csv', ''))
    climateList = list(set(climateList))
    return climateList

def getClimateName4Prototype(directory, prototype):
    # USE: get all the climate names for a prototype
    # INPUT: the dir of sim results
    climateList = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            prototypeGet = filename.split('____')[0]
            climateID = filename.split('____')[1].replace('.csv', '')
            if prototypeGet == prototype:
                climateList.append(climateID)
    return climateList

def getAllClimatesData(climates, addrWeather):
    # USE: get the data of all climates
    # INPUT: climates, list of climate names
    # INPUT: addrWeather, the dir of weather data
    
    data = importWeatherData(addrWeather, climates[0])
    for cli in climates[1:]:
        data_0 = importWeatherData(addrWeather, cli)
        data = pd.concat([data, data_0], axis = 0)
#     data.insert(0, 'Climate', data.pop('Climate'))
    return data

def getAllData4Prototype(proto, protoClimate, addrProto, addrWeather, addrTypical, col): # for adding the typical
    
    # read typical energy load/ heat emission
    dataTypical = importTypical(addrTypical, proto, col) # for adding the typical
    
    # initilization
    dataEnergy = importRawData(addrProto + '/' + proto + '____' + protoClimate[0] + '.csv', col = col)
    dataWeather = importWeatherData(addrWeather, protoClimate[0], len(dataEnergy))
#     data = pd.concat([dataEnergy, dataWeather], axis = 1)
    data = pd.concat([dataEnergy, dataWeather, dataTypical], axis = 1) # for adding the typical

    
    # loop through the rest of climates
    for cli in protoClimate[1:]:
        fileName = proto + '____' + cli + '.csv'
        dataEnergy = importRawData(addrProto + '/' + fileName, col = col)
        dataWeather = importWeatherData(addrWeather, cli, len(dataEnergy))
        
#         data_0 = pd.concat([dataEnergy, dataWeather], axis = 1)
        data_0 = pd.concat([dataEnergy, dataWeather, dataTypical], axis = 1) # for adding the typical
        
        data = pd.concat([data, data_0], axis = 0)
    data.insert(0, 'Climate', data.pop('Climate'))
    
    return data