"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-08-31

Contains model implementations.
"""


def _build_multistep_RNN(params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

    RNN = Sequential([
        LSTM(params['lstm_size'], input_shape=(None, 17)),
        Dense(params['len_prediction'], activation='relu')
    ])
    return RNN


def _build_seq2seq_regressor(params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

    RNN = Sequential([
        LSTM(params['len_input'], input_shape=(None, 17), return_sequences=True),
        LSTM(params['len_input'], return_sequences=True),
        TimeDistributed(Dense(1, activation='relu'))
    ])
    return RNN


def build(params):
    """
    [ Doc to be rewritten ]
    """
    if params['model_type'] == 1:
        model = _build_multistep_RNN(params)
    elif params['model_type'] == 2:
        model = _build_seq2seq_regressor(params)
    else:
        print('\nERROR: the model type specified in config.yaml is not valid.')
        print("Set:\n'model_type' = 1: plain multistep regressor\n'model_type' = 2: seq2seq regressor")
        quit()
    return model
