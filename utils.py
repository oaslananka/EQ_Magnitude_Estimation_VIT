import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler


def bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=100.0, order=5):
    """
    Applies a bandpass filter to the input data to remove noise outside the specified frequency range.

    Parameters:
    data (np.ndarray): Input data to filter
    lowcut (float): Low cutoff frequency
    highcut (float): High cutoff frequency
    fs (float): Sampling frequency
    order (int): Order of the filter

    Returns:
    np.ndarray: Filtered data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)


def normalize_data(data):
    """
    Normalizes the input data by subtracting the mean and dividing by the standard deviation.

    Parameters:
    data (np.ndarray): Input data to normalize

    Returns:
    np.ndarray: Normalized data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)
