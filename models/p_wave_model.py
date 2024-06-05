import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten


def create_p_wave_detection_model(input_shape):
    """
    Creates and compiles a convolutional neural network (CNN) model for P-wave detection.

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
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    input_shape = (6000, 3)
    p_wave_model = create_p_wave_detection_model(input_shape)
    p_wave_model.summary()
