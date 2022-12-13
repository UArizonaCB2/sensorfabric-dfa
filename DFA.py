import numpy as np
import pandas as pd

"""
Method which calculates the cumulative sum or the signal profile of the time series signal
"""
def calSignalProfile(x : np.array) -> np.array:
    mean = np.mean(x)
    x = x - mean
    return np.cumsum(x)

"""
Description
-----------
Method which detrends the given signal.

Parameters
-----------
signal : The signal to detrend. Usually a window of the signal profile (see calSignalProfile())
deg : The degree to use for fitting the trending function. Defaults to linear trend (deg = 1)


Returns
-------
standDeviation : Standard deviation of the detrended signal in the window.
trend : (np.array) The trend line.
detrended : (np.array) The detrended signal.
"""
def detrendSignal(signal : np.array, deg=1) -> (np.float64, np.array, np.array):
    if (not (type(signal) == type(np.empty(1)))):
        raise Exception('The passed y must be of type np.1darray')
    # Check if there are at least 4 elements inside the window.
    if signal.size <= 4:
        raise Exception('The window must have at least 4 elements')
        
    x = np.arange(0, signal.size, 1)
        
    # Calculate the linear trend.
    trend_coef = np.polyfit(x, signal, 1)
    trend = np.poly1d(trend_coef)
    # Calculate the linear trend signal
    T = trend(x)
    # Remove the linear trend to calculate the detrended signal.
    detrended = signal - T
    # calculate the standard deviation of the signal. 
    D = np.std(detrended)
    
    return D, T, detrended

"""
Given a set of window indexes (startindex : endindex), 
it will return a single detrended fluctuation value for them.
Uses the standard deviation from detrendingSignal() method as follows to calculate the fluctuation
fluctuation = mean([detrendedSignal(W1), detrendedSignal(W2), .....])

Parameters
----------
signal : The complete time series signal.
windows : A list of window indices [[start1:stop1], [start2:stop2], ...]]

Returns
-------
F : The Detrended fluctuation for the signal. 
"""
def calDetrendFlucuation(signal : np.array, windows : list) -> np.float64:
    # Will store the standard deviation returned by detrendSignal() on the windows.
    Dstd = np.empty(len(windows))
    for i,W in enumerate(windows):
        startIndex = W[0]
        stopIndex = W[1]
        
        # Get the segment of the signal.
        segment = signal[startIndex:stopIndex]
        # Get the standard deviation.
        Dstd[i], _, _ = detrendSignal(segment)
    
    # Calculate the fluctuation value.
    F = np.mean(Dstd)
    
    return F
