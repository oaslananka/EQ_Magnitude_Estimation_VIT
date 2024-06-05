import numpy as np
import tensorflow as tf
from utils import bandpass_filter, normalize_data
from models.p_wave_model import create_p_wave_detection_model
from models.magnitude_model import create_magnitude_estimation_model


def load_models():
    """
    Loads trained models from file.

    Returns:
    tuple: Loaded P-wave detection model and magnitude estimation model
    """
    p_wave_model = tf.keras.models.load_model('data/p_wave_model.h5')
    magnitude_model = tf.keras.models.load_model('data/magnitude_model.h5')
    return p_wave_model, magnitude_model


def simulate_real_time_data(samples=6000, devices=20):
    """
    Simulates real-time 3-axis accelerometer data for a specified number of devices and samples.

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


def process_real_time_data(device_data, p_wave_model, magnitude_model):
    """
    Processes real-time data to detect P-wave arrivals and estimate earthquake magnitude.

    Parameters:
    device_data (np.ndarray): Input data from a device
    p_wave_model (tensorflow.keras.models.Sequential): Trained P-wave detection model
    magnitude_model (tensorflow.keras.models.Sequential): Trained magnitude estimation model

    Returns:
    tuple: Detected P-wave arrival times and estimated magnitudes
    """
    filtered_data = bandpass_filter(device_data)
    normalized_data = normalize_data(filtered_data)

    p_wave_predictions = p_wave_model.predict(np.expand_dims(normalized_data, axis=0))
    p_wave_arrival_times = np.where(p_wave_predictions > 0.5)[1]  # Detect P-wave arrival times

    magnitudes = []
    for arrival_time in p_wave_arrival_times:
        # 30-second window (1 second before and 29 seconds after)
        if arrival_time >= 100 and arrival_time + 2900 < device_data.shape[0]:
            window = normalized_data[arrival_time-100:arrival_time+2900, :]
            magnitude_prediction = magnitude_model.predict(np.expand_dims(window, axis=0))
            magnitudes.append(magnitude_prediction)

    return p_wave_arrival_times, magnitudes


if __name__ == "__main__":
    p_wave_model, magnitude_model = load_models()
    real_time_data = simulate_real_time_data()

    for device_data in real_time_data:
        p_wave_times, magnitude_estimations = process_real_time_data(device_data, p_wave_model, magnitude_model)
        print(f"P-Wave Arrival Times: {p_wave_times}")
        print(f"Magnitude Estimations: {magnitude_estimations}")
