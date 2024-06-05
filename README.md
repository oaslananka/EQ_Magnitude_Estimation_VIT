
# EQ_Magnitude_Estimation_VIT

Great repository names are short and memorable. Need inspiration? How about **EQ_Magnitude_Estimation_VIT**?

**Description:**

This repository contains a comprehensive project for real-time earthquake detection and magnitude estimation using Vision Transformer (ViT). The project processes 3-axis accelerometer data from multiple devices, providing accurate P-wave detection and earthquake magnitude estimation in real-time.

## Key Features
- Real-time P-wave detection using a convolutional neural network (CNN) model.
- Accurate earthquake magnitude estimation with a Vision Transformer (ViT) model.
- Synthetic data generation, preprocessing, and normalization for training.
- Detailed model training scripts with early stopping to prevent overfitting.
- Real-time data processing and estimation with sample simulations.

## Folder Structure
- **data_preparation.py**: Script for generating and preprocessing synthetic accelerometer data.
- **models/**: Directory containing the model definitions for P-wave detection and magnitude estimation.
  - **p_wave_model.py**: P-wave detection model definition.
  - **magnitude_model.py**: Magnitude estimation model definition.
- **train_models.py**: Script for training both P-wave detection and magnitude estimation models.
- **real_time_detection.py**: Script for simulating real-time data and performing earthquake detection and magnitude estimation.
- **utils.py**: Utility functions for data preprocessing.
- **data/**: Directory for storing prepared data and trained model files.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/oaslananka/EQ_Magnitude_Estimation_VIT.git
   cd EQ_Magnitude_Estimation_VIT
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the data:
   ```bash
   python data_preparation.py
   ```

4. Train the models:
   ```bash
   python train_models.py
   ```

5. Simulate real-time detection:
   ```bash
   python real_time_detection.py
   ```

## License
This project is licensed under the MIT License.
