"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-08-31

Contains model implementations.
"""

def _build_singlestep_RNN(params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

    RNN = Sequential([
        LSTM(params['lstm_size'], input_shape=(None, 17)),
        Dense(params['len_prediction'], activation='relu', kernel_initializer=tf.keras.initializers.Zeros())
    ])
    return rnn_regressor


def _build_multistep_RNN(params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

    RNN = Sequential([
        LSTM(params['lstm_size'], input_shape=(None, 17)),
        Dense(params['len_prediction'], activation='relu', kernel_initializer=tf.keras.initializers.Zeros())
    ])
    return RNN


def _build_seq2seq_regressor(params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

    RNN = Sequential([
        LSTM(params['len_input'], input_shape=(None, 17), return_sequences=True),
        LSTM(params['len_input'], return_sequences=True),
        TimeDistributed(Dense(params['dense_size'], activation='relu', kernel_initializer=tf.keras.initializers.Zeros()))
    ])
    return RNN


def build(params):
    """
    [ Doc to be rewritten ]
    """
    model = _build_multistep_RNN(params)
    return model
