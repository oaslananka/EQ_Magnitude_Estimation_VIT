import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from models.p_wave_model import create_p_wave_detection_model
from models.magnitude_model import create_magnitude_estimation_model


def load_data():
    """
    Loads prepared data from file.

    Returns:
    np.ndarray: Loaded data
    """
    return np.load('data/prepared_data.npy')


def label_data(data):
    """
    Generates simulated labels for training. Replace with real labels in a production scenario.

    Parameters:
    data (np.ndarray): Input data to label

    Returns:
    tuple: P-wave labels and magnitude labels
    """
    # Generating one label per sample instead of one label per time step
    p_wave_labels = np.random.randint(0, 2, size=(data.shape[0], 1))
    magnitude_labels = np.random.uniform(0, 5, size=(data.shape[0],))
    return p_wave_labels, magnitude_labels


if __name__ == "__main__":
    data = load_data()
    p_wave_labels, magnitude_labels = label_data(data)

    # Split data into training and testing sets
    p_wave_train, p_wave_test, p_wave_labels_train, p_wave_labels_test = train_test_split(data, p_wave_labels, test_size=0.2, random_state=42)
    magnitude_train, magnitude_test, magnitude_labels_train, magnitude_labels_test = train_test_split(data, magnitude_labels, test_size=0.2, random_state=42)

    # Create models
    p_wave_model = create_p_wave_detection_model((6000, 3))
    magnitude_model = create_magnitude_estimation_model((3000, 3))

    # Early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train P-wave detection model
    p_wave_model.fit(p_wave_train, p_wave_labels_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
    p_wave_model.save('data/p_wave_model.h5')

    # Train magnitude estimation model
    magnitude_model.fit(magnitude_train[:, 100:3100, :], magnitude_labels_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
    magnitude_model.save('data/magnitude_model.h5')
