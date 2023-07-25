import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.style as style
from scipy.linalg import lstsq
from scipy.stats import pearsonr

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
    
def plotPCC(data, weatherFeatures, target, maxLag, save):
    # # partial cross correlation
    fig = go.Figure()
    for weatherFeature in weatherFeatures:
        pcc = partial_xcorr(data[weatherFeature].values,
                            data[target].values,
                            max_lag = maxLag)
        fig.add_scatter(x = np.arange(pcc.shape[0]), y = pcc, mode = 'lines', name = weatherFeature)
    fig.update_layout(xaxis_title = 'Lags',
                      yaxis_title = 'Partial Cross Correlation',
                     )
    fig.write_html(save)
    
def plotPeaks(data, target):
    style.use('seaborn')

    peaks = find_peaks(data[target].values, prominence = 1)[0]

    fig, ax = plt.subplots(figsize = (7, 5))
    ax.plot(data[target].values, label = 'Original data', lw = 0.75)
    ax.plot(peaks, data[target].values[peaks], 'o', label = 'Peaks', )
    ax.legend()

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

def partial_xcorr(x, y, max_lag = 10, standardize = True):
    # Computes partial cross correlation between x and y using linear regression.
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

def partial_xcorr_reverse(x, y, max_lag = 10, standardize = True):
    # Computes partial cross correlation between x and y using linear regression.
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
        X[:, lag] = _shift(x, -lag)
        # Trim NaNs from y (yt), current lag (l) and previous lags (Z)
        yt = y[:-lag]
        l = X[:-lag, lag: lag + 1] # this time lag
        Z = X[:-lag, 0: lag] # all previouse time lags
        # Coefficients and residuals for y ~ Z
        beta_l = lstsq(Z, yt)[0]
        resid_l = yt - Z.dot(beta_l)
        # Coefficient and residuals for l ~ Z
        beta_Z = lstsq(Z, l)[0]
        resid_Z = l - Z.dot(beta_Z)
        # Compute correlation between residuals
        xcorr[lag] = pearsonr(resid_l, resid_Z.ravel())[0]
    return xcorr