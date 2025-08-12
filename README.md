# Time Series Forecasting with LSTM

This project implements a time series forecasting model using a Long Short-Term Memory (LSTM) neural network with PyTorch. The model is trained to predict the number of airline passengers based on historical data.

## 🚀 Getting Started

This repository contains a Jupyter Notebook that guides you through the process of building and evaluating an LSTM model for time series forecasting. The notebook covers data loading, preprocessing, model definition, training, and visualization of the results.

### Prerequisites

To run this project, you'll need to have Python and the following libraries installed. You can install them using `pip`:

```bash
pip install torch numpy pandas seaborn matplotlib scikit-learn
```

### Usage

1.  **Clone the repository** (if it's not already cloned).
2.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook timeserie_forcasting_hwlast.ipynb
    ```
3.  **Run the cells in order** to execute the entire workflow, from data preparation to model evaluation.

## 📈 Project Overview

### Dataset

The project uses the `flights` dataset from the `seaborn` library. This dataset contains monthly passenger numbers from 1949 to 1960.

### Model

  - **Model Architecture**: A custom LSTM model is defined using PyTorch (`nn.Module`). It consists of an LSTM layer followed by a linear layer.
  - **Input**: The model takes a sequence of historical passenger numbers to predict the next value in the series. The `window` size is set to 4, meaning the model uses the last 4 months of data to predict the next month's value.
  - **Training**:
      - **Loss Function**: Mean Squared Error (`nn.MSELoss`)
      - **Optimizer**: Stochastic Gradient Descent (`torch.optim.SGD`) with a learning rate of `0.001`
      - **Epochs**: The model is trained for `500` epochs.

## 📊 Results

The final plot shows the model's predictions on the test data compared to the actual values. This visualization demonstrates how well the LSTM model captures the underlying trends and seasonality of the passenger data.
