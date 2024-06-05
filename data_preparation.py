import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(samples=6000, devices=20):
    """
    Generates synthetic 3-axis accelerometer data for a specified number of devices and samples.
    Each device generates data independently with normal distribution noise.

    Parameters:
    samples (int): Number of samples per device
    devices (int): Number of devices

    Returns:
    np.ndarray: Array of shape (devices, samples, 3)
    """
    data = []
    for _ in range(devices):
        x = np.random.normal(0, 1, samples)
        y = np.random.normal(0, 1, samples)
        z = np.random.normal(0, 1, samples)
        data.append(np.stack([x, y, z], axis=1))
    return np.array(data)


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


def prepare_data():
    """
    Prepares the data by generating synthetic data, applying bandpass filter, and normalizing.

    Returns:
    np.ndarray: Prepared data
    """
    raw_data = generate_synthetic_data()
    filtered_data = np.array([bandpass_filter(device_data) for device_data in raw_data])
    normalized_data = np.array([normalize_data(device_data) for device_data in filtered_data])
    return normalized_data


if __name__ == "__main__":
    prepared_data = prepare_data()
    np.save('data/prepared_data.npy', prepared_data)
