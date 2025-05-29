# Human Activity Recognition (HAR) with MotionSense Dataset

![HAR System Overview](https://img.shields.io/badge/Field-Computer_Vision-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning pipeline for Human Activity Recognition using smartphone sensor data from the MotionSense dataset.

## Project Overview
This repository contains:
- **Dataset**: MotionSense (raw IMU sensor data: accelerometer, gyroscope, attitude)
- **Model**: 1D CNN implemented in PyTorch
- **Input Shape**: (batch_size, channels, timesteps) = (N, 6, 100)
- **Windowing Strategy**: Fixed-size sliding windows (100 timesteps, 50% overlap)
- **Activities**: Walk, Jog, Sit, Stand, Upstairs, Downstairs (subset selectable)
- **Evaluation**: Accuracy, Confusion Matrix, F1 Score

  
## Dataset
The project uses the **[MotionSense Dataset](https://github.com/mmalekzadeh/motion-sense)**:
- Collected from 24 participants using iPhone 6s
- 6 activities: Walking, Jogging, Sitting, Standing, Stair Ascent, Stair Descent
- 50Hz sampling rate from accelerometer and gyroscope
- Device placement: Front pocket (real-world conditions)

>**Reference**:  
> Malekzadeh et al. (2019). ["Mobile Sensor Data Anonymization"](https://dl.acm.org/doi/10.1145/3302505.3310068). *Proceedings of IoTDI*.


## Results

| Model | Accuracy | Latency | Parameters |
|-------|----------|---------|------------|
| 1D CNN | 93.31% | 42ms | 1.2M |

---
## Model Architecture

```text
Input (100 x 6)
↓
Conv1D (64 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
↓
Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
↓
Flatten
↓
Dense(128) + ReLU + Dropout(0.5)
↓
Dense(6) + Softmax

