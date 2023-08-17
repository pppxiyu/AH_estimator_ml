import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.style as style
from scipy.linalg import lstsq
from scipy.stats import pearsonr
import pandas as pd
from utils.eval import cv_mean_absolute_error_wAbs


######################## vis: raw data ############

def plotPACF(acfs, save):
    lowerB = acfs[1][:,0] - acfs[0]
    upperB = acfs[1][:,1] - acfs[0]

    fig = go.Figure()
    [fig.add_scatter(x = (x,x), y = (0, acfs[0][x]), mode = 'lines', line_color='#3f3f3f') for x in range(len(acfs[0]))]
    fig.add_scatter(x = np.arange(len(acfs[0])), y = acfs[0], mode = 'markers', marker_color = '#1f77b4', marker_size = 8)
    fig.add_scatter(x = np.arange(len(acfs[0])), y = upperB, mode = 'lines', line_color = 'rgba(255,255,255,0)')
    fig.add_scatter(x = np.arange(len(acfs[0])), y = lowerB, mode = 'lines',fillcolor = 'rgba(32,146,230,0.3)', fill = 'tonexty', line_color = 'rgba(255,255,255,0)')

    fig.update_traces(showlegend = False)
    fig.update_yaxes(zerolinecolor = '#000000')
    fig.write_html(save)


def plotPCC(data, weatherFeatures, target, maxLag, save,
            reverse = False,
            xaxisTitle = 'Lags',
            ):
    # # partial cross correlation
    fig = go.Figure()
    for weatherFeature in weatherFeatures:
        pcc = partial_xcorr(data[weatherFeature].values,
                            data[target].values,
                            max_lag = maxLag,
                            reverse = reverse,
                            )
        fig.add_scatter(x = np.arange(pcc.shape[0]), y = pcc, mode = 'lines', name = weatherFeature)
    fig.update_layout(xaxis_title = xaxisTitle,
                      yaxis_title = 'Partial Cross Correlation',
                      template = 'seaborn',
                      yaxis_range=[-0.6, 0.6],
                      xaxis_autorange = 'reversed',
                      # yaxis_showticklabels = False,
                      font_size = 16
                     )
    fig.write_html(save)


def plotPeaks(data, target):
    style.use('seaborn')
    peaks = find_peaks(data[target].values, prominence = 1)[0]
    fig, ax = plt.subplots(figsize = (7, 5))
    ax.plot(data[target].values, label = 'Original data', lw = 0.75)
    ax.plot(peaks, data[target].values[peaks], 'o', label = 'Peaks', )
    ax.legend()
    plt.show()
    return

def plotPeaks_plotly(data, target, addrSave):
    # USE: draw the time series and peaks

    # clean data
    data['timestamp'] = data['timestamp'].str.replace('2001', '2018')
    data = data.set_index('timestamp')
    # get peaks
    promiOverMean_peak = 1.276
    peaks = find_peaks(data[target].values,
                       prominence = promiOverMean_peak * data.true.mean()
                       )[0]
    # get valleys
    promiOverMean_valley = 0.7
    valleys = find_peaks(-data[target].values,
                       prominence = promiOverMean_valley * data.true.mean()
                       )[0]
    # vis
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = data.index,
        y = data[target],
        mode = 'lines',
        name = 'Electricity Consumption (kWh)',
        showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = data.index[peaks],
        y = [data[target][j] for j in peaks],
        mode = 'markers',
        marker = dict(
            size = 6,
        ),
        name = 'Peaks'
    )),
    fig.add_trace(go.Scatter(
        x = data.index[valleys],
        y = [data[target][j] for j in valleys],
        mode = 'markers',
        marker = dict(
            size = 6,
        ),
        name = 'Valleys'
    )),
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'Time (hour)',
                      yaxis_title = 'Electricity Consumption (kWh)',
                      )
    fig.write_html(addrSave)
    return

def _shift(x, by):
    # if by > 0, nan will be at begining, first non-nan value was the first value
    # vice versa
    x_shift = np.empty_like(x)
    if by > 0:
        x_shift[:by] = np.nan
        x_shift[by:] = x[:-by]
    else:
        x_shift[by:] = np.nan
        x_shift[:by] = x[-by:]
    return x_shift


def partial_xcorr(x, y, max_lag = 10, standardize = True, reverse = False):
    # Computes partial cross correlation between x and y using linear regression.
    if reverse == True:
        x = x[::-1]
        y = y[::-1]
    if standardize:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    # Initialize feature matrix
    nlags = max_lag + 1
    X = np.zeros((len(x), nlags))
    X[:, 0] = x
    # Initialize correlations, first is y ~ x
    xcorr = np.zeros(nlags, dtype = float)
    xcorr[0] = pearsonr(x, y)[0]
    # Process lags
    for lag in range(1, nlags):
        # Add lag to matrix
        X[:, lag] = _shift(x, lag)
        # Trim NaNs from y (yt), current lag (l) and previous lags (Z)
        yt = y[lag:]
        l = X[lag:, lag: lag + 1] # this time lag
        Z = X[lag:, 0: lag] # all previouse time lags
        # Coefficients and residuals for y ~ Z
        beta_l = lstsq(Z, yt)[0]
        resid_l = yt - Z.dot(beta_l)
        # Coefficient and residuals for l ~ Z
        beta_Z = lstsq(Z, l)[0]
        resid_Z = l - Z.dot(beta_Z)
        # Compute correlation between residuals
        xcorr[lag] = pearsonr(resid_l, resid_Z.ravel())[0]
    return xcorr


######################## vis: result data ############
def plotPrototypeLevelMetrics_plotly(predPrototypeLevel, metricName, colorList, metricFunc, addr):
    # USE: draw the metrics plot at prototype level
    #      using plotly
    # INPUT: predPrototypeLevel, metricName, colorList: list
    #        metricFunc: function object
    #        addr: string
    # OUTPUT: save plot to the folder

    fig = go.Figure()
    for pred, name, color in zip(predPrototypeLevel, metricName, colorList):
        metric_prototype_ave_dict = metricPrototype(metricPrototypeWeather(pred, metricFunc), name)

        prototypes = metric_prototype_ave_dict['prototype'].tolist()

        fig.add_trace(go.Bar(x = metric_prototype_ave_dict[name].tolist(),
                             y = prototypes,
                             name = name.split('_')[1],
                             marker_color = color,
                             orientation = 'h',
                             ))
    fig.update_layout(
        yaxis = dict(
            title = 'Prototype',
            titlefont_size = 14,
            tickfont_size = 11,
        ),
        xaxis = dict(
            title = metricName[0].split('_')[0],
            titlefont_size = 14,
            tickfont_size = 12,
        ),
        legend=dict(
            x = 0.7,
            y = 0,
            bgcolor = 'rgba(255, 255, 255, 0)',
            bordercolor = 'rgba(255, 255, 255, 0)'
        ),
        barmode = 'group',
        bargap = 0.1,  # gap between bars of adjacent location coordinates.
        bargroupgap = 0.05  # gap between bars of the same location coordinate.
    )
    fig.write_html(addr + '/' + metricName[0].split('_')[0] + '_prototype.html')

    print('Prototype level metric fig saved.')

    return

def plotPrototypeLevelLines_plotly(protoDf, saveDir, title = None):
    # USE: draw the line chart, one line is grouth truth, another is estimates

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = protoDf['DateTime'], y = protoDf['true'], name = "True",
                             line_shape='linear'))
    fig.add_trace(go.Scatter(x = protoDf['DateTime'], y = protoDf['estimate'], name = "Estimate",
                             line_shape='linear'))
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'Time (hour)',
                      yaxis_title = 'Electricity Consumption (kWh)',
                      title = title
                      )
    fig.write_html(saveDir)
    return


def plotTractLevelPredictionLine(df, tractSelect, addr, selectIndex):
    # USE: draw the prediction line graph for one census tract
    # INPUT: the tract level df, containing both true and estimate
    #        census tract name
    #        the dir of the experiment folder
    # OUTPUT: save plot to the folder

    dfSelect = df[df.geoid == tractSelect]
    dfSelect = dfSelect.reset_index()
    dfSelect = dfSelect.loc[selectIndex[0]: selectIndex[1]]
    print(cv_mean_absolute_error_wAbs(dfSelect.true, dfSelect.estimate))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dfSelect.timestamp,
                             y = dfSelect.true,
                             name = "true",
                             line_shape = 'linear'))
    fig.add_trace(go.Scatter(x = dfSelect.timestamp,
                             y = dfSelect.estimate,
                             name = "estimate",
                             line_shape = 'linear'))
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'Time (hour)',
                      yaxis_title = 'Electricity Consumption (kWh)',
                      )
    # fig.write_html(addr + '/' + 'tractEstimation_' + str(tractSelect) + '.html')
    fig.write_html(addr)

    return


def getTractIndexinWeatherZone(experimentLabel, selectedClimate, metaAddr):
    # USE: given a selected list of weathers, output tracts that are predicted by model

    # get tracts in selected climates
    buildingMeta = pd.read_csv(metaAddr).groupby(
        ['id.tract', 'id.grid.coarse']).count()
    buildingMeta = buildingMeta.reset_index()[['id.tract', 'id.grid.coarse']]
    buildingMeta = buildingMeta[buildingMeta['id.grid.coarse'].isin(selectedClimate)]
    # get predictions in the selected climates
    predTractLevel = pd.read_csv('./saved/estimates_tracts/' + experimentLabel + '/tractsDF.csv')
    predTractLevel = predTractLevel.merge(buildingMeta, how='left', left_on='geoid', right_on='id.tract')
    predTractLevel = predTractLevel[~predTractLevel['id.grid.coarse'].isna()]
    for cli in selectedClimate:
        tractList = predTractLevel[predTractLevel['id.grid.coarse'] == cli]['id.tract'].unique()
        tractList = [str(int(i)) for i in tractList]
        print('Tract index in ' + str(cli) + ' is' + str(tractList))

    return


def plotTractLevelCVMAELine_wShadow(predTractLevel, selectIndex, addrSave):
    # USE: plot the CVMAE time series (true and estimate) in the year
    #      becasue there are many tracts, the plot is mean line with error bar

    if (predTractLevel['true'] <= 0).any():
        print('Negative value or zero exist!')
        return
    predTractLevel['CVMAE'] = np.abs(predTractLevel.true.values - predTractLevel.estimate.values) / predTractLevel.true.values
    predTractMean = predTractLevel.groupby('timestamp').mean().reset_index()
    predTractMean = predTractMean[['timestamp', 'CVMAE']]
    predTractMean = predTractMean.rename(columns={'CVMAE': 'CVMAE_mean'})
    predTractStd = predTractLevel.groupby('timestamp').std().reset_index(drop=True)
    predTractStd = predTractStd[['CVMAE']]
    predTractStd = predTractStd.rename(columns={'CVMAE': 'CVMAE_std'})
    predTractReady = pd.concat([predTractMean, predTractStd], axis = 1)

    predTractReady['timestamp'] = predTractReady['timestamp'].str.replace('2001', '2018')
    predTractReady = predTractReady.iloc[selectIndex[0]: selectIndex[1]]

    fig = go.Figure([
        # true
        go.Scatter(
            name = 'CVMAE',
            x = predTractReady['timestamp'],
            y = predTractReady['CVMAE_mean'],
            mode = 'lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name = 'Upper Bound',
            x = predTractReady['timestamp'],
            y = predTractReady['CVMAE_mean'] + predTractReady['CVMAE_std'],
            mode = 'lines',
            line=dict(width=0),
            # marker = dict(color="#444"),
            showlegend = False
        ),
        go.Scatter(
            name='Lower Bound',
            x = predTractReady['timestamp'],
            y = predTractReady['CVMAE_mean'] - predTractReady['CVMAE_std'],
            # marker = dict(color = "#444"),
            mode = 'lines',
            line=dict(width=0),
            fillcolor = 'rgba(68, 68, 68, 0.3)',
            fill = 'tonexty',
            showlegend = False
        ),
    ])
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'Time (hour)',
                      yaxis_title = 'nMAE',
                      yaxis_range = [-0.2, 0.6],
                      )
    fig.write_html(addrSave)
    return


def plotTractLevelTargetEstimate_wShadow(predTractLevel, trueEstimate, selectIndex, addrSave):
    # USE: plot the ntarger time series (true and estimate) in the year
    #      becasue there are many tracts, the plot is mean line with error bar

    predTractMean = predTractLevel.groupby('timestamp').mean().reset_index()
    predTractMean = predTractMean[['timestamp', trueEstimate]]
    predTractMean = predTractMean.rename(columns={trueEstimate: trueEstimate + '_mean'})
    predTractStd = predTractLevel.groupby('timestamp').std().reset_index(drop=True)
    predTractStd = predTractStd[[trueEstimate]]
    predTractStd = predTractStd.rename(columns={trueEstimate: trueEstimate + '_std'})
    predTractReady = pd.concat([predTractMean, predTractStd], axis = 1)

    predTractReady['timestamp'] = predTractReady['timestamp'].str.replace('2001', '2018')
    predTractReady = predTractReady.iloc[selectIndex[0]: selectIndex[1]]

    fig = go.Figure([
        go.Scatter(
            name = trueEstimate,
            x = predTractReady['timestamp'],
            y = predTractReady[trueEstimate + '_mean'],
            mode = 'lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
    ])
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'Time (hour)',
                      yaxis_title = 'Electricity (kWh)',
                      yaxis_range = [0, 6000]
                      )
    fig.write_html(addrSave)
    return


def plotDistPeakCVMAE(data, addrSave):
    # USE: draw the CVMAE dist of tracts on peaks and on non-peaks

    # funcs
    promiOverMean_peaks = 1.276
    promiOverMean_valleys = 0.7
    def getCVAME_peaks(true, estimate):
        peaks = find_peaks(true.values, prominence = promiOverMean_peaks * true.mean())[0]
        return cv_mean_absolute_error_wAbs(true.iloc[peaks].values, estimate.iloc[peaks].values)
    def getCVAME_valleys(true, estimate):
        valleys = find_peaks(-true.values, prominence = promiOverMean_valleys * true.mean())[0]
        return cv_mean_absolute_error_wAbs(true.iloc[valleys].values, estimate.iloc[valleys].values)
    def getCVAME_nonPeaks(true, estimate):
        peaks = find_peaks(true.values, prominence = promiOverMean_peaks * true.mean())[0]
        valleys = find_peaks(-true.values, prominence=promiOverMean_valleys * true.mean())[0]
        return cv_mean_absolute_error_wAbs(true.iloc[~np.concatenate([peaks, valleys])].values,
                                           estimate.iloc[~np.concatenate([peaks, valleys])].values)

    # clean data
    data['timestamp'] = data['timestamp'].str.replace('2001', '2018')
    # group by tract and get CVMAE
    CVMAE_peaks = data.groupby('geoid').apply(lambda x: getCVAME_peaks(x['true'], x['estimate'])).to_frame()
    CVMAE_peaks = CVMAE_peaks.rename(columns = {0: "CVMAE_peaks"})
    CVMAE_valleys = data.groupby('geoid').apply(lambda x: getCVAME_valleys(x['true'], x['estimate'])).to_frame()
    CVMAE_valleys = CVMAE_valleys.rename(columns={0: "CVMAE_valleys"})
    CVMAE_non = data.groupby('geoid').apply(lambda x: getCVAME_nonPeaks(x['true'], x['estimate'])).to_frame()
    CVMAE_non = CVMAE_non.rename(columns={0: 'CVMAE_non'})

    CVMAE = pd.concat([CVMAE_peaks, CVMAE_valleys, CVMAE_non], axis = 1)

    fig = ff.create_distplot([CVMAE[c] for c in CVMAE.columns],
                             bin_size=.002,
                             show_rug = False,
                             group_labels = ['nMAE: peaks', 'nMAE: valleys', 'nMAE: rest of time'],
                             )
    fig.update_layout(template = 'seaborn',
                      font_size = 16,
                      xaxis_title = 'nMAE',
                      )
    fig.write_html(addrSave)
    return
