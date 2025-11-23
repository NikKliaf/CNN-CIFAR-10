# CIFAR-10 Image Classification using PyTorch
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## Project Description
This project implements and compares various methods for classifying images from the **CIFAR-10** dataset. The primary objective was to develop a **Deep Convolutional Neural Network** capable of outperforming classical machine learning classification algorithms

The project evolved through three phases:
1. **Baseline:** Implementation of distance-based classifiers (Nearest Neighbor & Nearest Class Centroid).
2. **Initial CNN:** A shallow network (2 convolutional layers) using simple convolutions.
3. **Optimized CNN:** A deep network (6 convolutional layers) utilizing Batch Normalization and Global Average Pooling.

## Model Architecture

The final optimized model follows a VGG-style architecture with the following structure:

* **Input:** $32 \times 32 \times 3$
* **Feature Extraction:** 6 Convolutional Layers. Each layer follows the pattern:
    * `Conv2d` -> `BatchNorm2d` -> `ReLU`
    * Progressive filter increase: $64 \rightarrow 128 \rightarrow 256$.

* **Classifier:**
    * `AdaptiveAvgPool2d(1)`: Replaces massive Flatten layers to reduce parameter count.
    * `Dropout(0.5)`: To prevent overfitting.
    * `Linear`: Final classification layer for 10 classes.
<br>

<img width="800" height="600" alt="cnn_architecture" src="https://github.com/user-attachments/assets/493104b4-6494-46ff-a2fb-14ca6acc4e7e" />

<br>
 
## Results & Comparison

A comparative performance table of the implemented methods:

| Method | Type | Accuracy | Observations |
| :--- | :--- | :--- | :--- |
| **NCC** (Nearest Class Centroid) | Distance-based | ~28% | Too simplistic, high bias. |
| **KNN** (k-Nearest Neighbors) | Distance-based | ~35% | Slow inference, poor generalization on pixel data. |
| **Simple CNN** | Deep Learning | ~65-70% | Prone to overfitting. |
| **Optimized CNN (Final)** | **Deep Learning** | **>80%** | **High generalization, fast convergence.** |

## Technical Details

* **Framework:** PyTorch & Torchvision
* **Optimizer:** SGD (Learning rate: 0.1, Weight Decay: 1e-4)
* **Loss Function:** Cross Entropy Loss
* **Hardware:** Training accelerated on **GPU** using **CUDA** (Training is too slow otherwise).

## Installation & Usage

1.  **Download the repository:**
    Download or clone the repository.
    ```bash
    git clone [https://github.com/NikKliaf/CNN-CIFAR-10.git](https://github.com/NikKliaf/CNN-CIFAR-10.git)
    cd CNN-CIFAR-10
    ```
    
2.  **Download Dependencies:**
    ```bash
    pip install torch torchvision matplotlib pillow numpy
    ```
3. **Optional: Install CUDA drivers for NVIDIA GPU:** <br>
    [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
    
4. **Run the Code:** <br>
    Run .ipynb files using [Jupyter Notebook](https://jupyter.org/install), [Google Colab](https://colab.research.google.com/) or [VSCode](https://code.visualstudio.com/).
   
## Author
Nikiforos Kliafas

Computer Science student at Aristotle University of Thessaloniki
