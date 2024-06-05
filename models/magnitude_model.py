import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten


def create_magnitude_estimation_model(input_shape):
    """
    Creates and compiles a convolutional neural network (CNN) model for earthquake magnitude estimation.

    Parameters:
    input_shape (tuple): Shape of the input data

    Returns:
    tensorflow.keras.models.Sequential: Compiled CNN model
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Conv1D(256, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


if __name__ == "__main__":
    input_shape = (3000, 3)
    magnitude_model = create_magnitude_estimation_model(input_shape)
    magnitude_model.summary()
