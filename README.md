# WORK IN PROGRESS

# Recurrent Neural Network for Time Series Forecasting

## Description
The model is a Recurrent Neural Network for time series forecast. It is implemented in **TensorFlow 2** and trained on the Wikipedia [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset from Kaggle.

## Model Structure
For a detailed explanation of its implementation, see [how_it_works.md]() file.

## Structure of the repository
Main files:
- `config.yaml`: config file for hyperparameters.
- `dataprep.py`: data preprocessing pipeline.
- `train.py`: training pipeline.
- `tools.py`: contains useful processing functions to be iterated in main pipelines.
- `model.py`: builds model.

Folders:
- `/data_raw/`: requires unzipped `train_2.csv` file from [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/). Available is an `imputed.csv` dataset, containing imputed time series, coming from my other repository on a [GAN for imputation of missing data in time series](https://github.com/IvanBongiorni/GAN-RNN_Timeseries-imputation).
- `/data_processed/`: divided in `/Train/` and `/Test/` directories.
- `/saved_models/`: contains all saved TensorFlow models, both regressors

## How to run code
After you clone the repository locally, download the raw dataset from [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/), and place unzipped `train_2.csv` file in `/data_raw/` folder.
Then, time series forecast is executed in two steps. First, run data preprocessing pipeline:

`python -m dataprep`

This will generate Trainging+Validation and Test files, stored in `/data_processed/` subdirectories. Second, launch training pipeline with:

`python -m train`

This will either create, train and save a new model, or load and train an already existing one, stored in `/saved_models/` folder.

Finally, Test set performance will be evaluated from `test.ipynb` notebook.


## Modules
```
numpy==1.18.3
pandas==1.0.3
scikit-learn==0.22.2.post1
scipy==1.4.1
tensorflow==2.1.0
```

## Hardware
I used a pretty powerful laptop, with 64GB or RAM and NVidia RTX 2070 GPU. I highly recommend GPU training to avoid excessive computational times.
