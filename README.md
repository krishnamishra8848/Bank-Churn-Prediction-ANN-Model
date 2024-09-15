# Customer Churn Prediction Model

This project implements a neural network model to predict customer churn. The model is built using TensorFlow and Keras for binary classification tasks.

## Model Architecture

- **Input Layer**: Handles preprocessed numerical and categorical features.
- **Hidden Layer**:
  - **Neurons**: 32
  - **Activation**: ReLU
  - **Initialization**: He normal
- **Output Layer**:
  - **Neurons**: 1
  - **Activation**: Sigmoid

## Key Features

- **Preprocessing**:
  - Numerical features are standardized.
  - Categorical features are one-hot encoded.
- **Training**:
  - **Optimizer**: Adam
  - **Loss Function**: Binary Crossentropy
  - **Metrics**: Accuracy

