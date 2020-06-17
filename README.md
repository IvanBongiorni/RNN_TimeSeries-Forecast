# WORK IN PROGRESS

# Convolutional Recurrent Seq2seq Model for Wikipedia Web Traffic Forecasting

## Description
The model is a Convolutional-Recurrent Neural Network for time series forecast. It is implemented in **TensorFlow 2.1** and trained on the Wikipedia [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset from Kaggle.

## Model Structure
For a detailed explanation of its implementation, see [how_it_works.md]() file.

## Structure of the repository
- `config.yaml`: configuration file for hyperparameters
- `input_pipe.py`: main preprocessing script
- `tools.py`: contains preprocessing functions
- `model.py`: contains model building, training and testing functions
- `/data/` folder: contains training data, downloadable from Kaggle
- `/imputer/` folder: contains imputation model, i.e. a pretrained Neural Network used in dataprep phase in order to impute missing values. The model is coming from my other repository (ADD LINK)

## How to run code
After you clone the repository locally, the whole code runs in two steps. The first is a pipeline that takes raw data and processes them to be fed into the model; it is activated by running

`python -m input_pipe.py`

from terminal. The second part is the actual training code; it is activated from terminal with

`python -m train_model.py`

## Modules
```
tensorflow == 2.1.0
numpy
pandas
```

## Hardware
I used a pretty powerful laptop, with 64GB or RAM and NVidia RTX 2070 GPU. I highly recommend GPU training to avoid excessive computational times.
