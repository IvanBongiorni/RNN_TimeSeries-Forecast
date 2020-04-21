# Seq2seq__Wikipedia_Web-Traffic_Forecast

The model is hybrid Seq2seq with LSTM and Convolutional Encoder. It is implemented in TensorFlow 2.1 and trained on the Wikipedia [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset from Kaggle.

## Structure of the repository
- config.yaml: configuration file for hyperparameters
- input_pipe.py: main preprocessing script
- tools.py: contains preprocessing functions
- model.py: contains model building, training and testing functions
- /data/ folder: contains training data, downloadable from Kaggle
- /imputer/ folder: contains imputation model, i.e. a pretrained Neural Network used in dataprep phase in order to impute missing values. The model is coming from my other repository (ADD LINK)
