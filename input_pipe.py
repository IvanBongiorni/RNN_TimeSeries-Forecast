"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-04-10

DATA PREPROCESSING PIPELINE. Takes raw data a process them for model training.

This script contains high level operations, i.e. processing steps for the whole
dataset of its parts. Operations on cell values are called from tools.py script.
"""


# TODO: assicurati che in load_dataframe() avvenga l'ordinamento per colonne


def load_dataframe(path_to_data):

    df = pd.read_csv(path_to_data + 'train_2.csv')

    # drop rows that are all NaN's
    df.dropna(axis = 'rows', how = 'all', inplace = True)

    # discard Test block on the rightdf
    test_cutoff = int(params['val_test_size'][1] * df.shape[1])
    df = df.iloc[ : , :test_cutoff ]
    return df


def get_time_schema(df):
    """ Returns np.array with patterns for time-related variables (year/week days)
    in [0,1] range, to be repeated on all trends. """
    daterange = pd.date_range(df.columns[1], df.columns[-1], freq='D').to_series()

    weekdays = daterange.dt.dayofweek
    weekdays = weekdays.values / weekdays.max()
    yeardays = daterange.dt.dayofyear
    yeardays = yeardays.values / yeardays.max()

    weekdays = weekdays.values
    yeardays = yeardays.values

    # First year won't enter the Train set because of year lag
    weekdays = weekdays[ 365: ]
    yeardays = yeardays[ 365: ]

    return weekdays, yeardays


def process_page_data(df):
    """
    Attaches page data to main df by iterating process_url() from
    dataprep_tools.py:
        language:  code - with 'na' for 'no language detected'
        website:   what type of website: 'wikipedia', 'wikimedia', 'mediawiki'
        access:    type of access (e.g.: mobile, desktop, both, ...)
        agent:     type of agent
    """
    page_data = [ process_url(url) for url in df['Page'].tolist() ]
    page_data = pd.concat(page_data, axis = 0)
    page_data.reset_index(drop = True, inplace = True)

    page_data['Page'] = df['Page'].copy()  # Attach 'Page' to page_data for merging

    # Attach page_data to main df and drops 'Page' col with links
    df = df.merge(page_data, on = 'Page')
    df.drop('Page', axis = 1, inplace = True)
    return df


def process_and_load_data():
    """
    Main wrapper for the whole pipe. This object is to be instantiated for loading
    and preprocessing the dataset cleanly from Jupyter Notebook or other scripts.
    """
    import os
    import yaml
    import time
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    import tools  # local module

    languages = [ 'en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na' ]

    print('Loading raw data and main hyperparams.')
    current_path = os.getcwd()
    df = load_dataframe(current_path + '/data/train2.csv')

    params = yaml.load(open(current_path + '/config.yaml'), yaml.Loader)

    print('Processing URL information.')
    page_data = process_page_data(page_data)

    print('Creating time schema')
    weekdays, yeardays = get_time_schema(df)

    print('Loading imputation model: {}'.format(params['imputation_model']))
    model = tf.keras.models.load_model(current_path + '/saved_models/' + params['imputation_model'])

    X_train = []
    scaling_dict = {}

    print('Preprocessing trends by language group.')
    for language in languages:
        start = time.time()

        sdf = df[ page_data['language'] == language ].values
        sdf_page_data = page_data[ page_data['language'] == language ].values

        print('\t{}'.format(language))
        for i in range(sdf.shape[0]):
            X_train.append( RNN_dataprep(t = sdf[i,:],
                                         page_vars = sdf_page_data[i,:],
                                         day_week = weekdays,
                                         day_year = yeardays,
                                         params = params))

        ### TODO: La scalatura deve avvenire dopo l'imputazione (per evitare i NaN)
        #   e tenendo fuori i dati di validation


        ### IMPORTANTE: Correggere scale_trends() applicando la scalatura sui dati di Train
        sdf, scaling_percentile, = scale_trends(sdf, params)
        scaling_dict[language] = scaling_percentile

        print('\t{} done in {} ss.'.format(language, round(time.time()-start, 2)))

    ### TODO: Salvare il dizionario per la scalatura

    # Pickle scaling params once it's done

    print('Loading imputation model: {}.h5'.format(params['imputation_model']))

    imputer = tf.keras.load_model('{}/imputation_model/{}.h5'.format(current_path, params['imputation_model']))

    # Order of variables is:
        # t,
        # trend_lag_quarter,
        # trend_lag_year,
        # weekdays,
        # yeardays
        # + page variables

    # I must run imputation model on the first three variables (i.e. trend data)
    for i in range(3):
        original = X_train(X_train[ : , : , i ])
        imputed = imputer.predict(original)

        # Substitute imputed to original only when NaN and put it back
        original[ np.isnan(original) ] = imputed
        X_train[ : , : , i ] = original

    del original, imputed, imputer

    # Split in Train and Validation
    cut = int()

    #####
    # TODO: ESEGUIRE PARTIZIONE TRAIN - VAL

    Y_train = X_train[ : , cut: , : ]

    # df = attach_page_data(df)
    # # Prepares data matrices ready for Jupyter Notebooks
    # X_final, Y_final = get_train_and_target_data(df, len_test_series = params['len_test_series'])



    return X_train, Y_train, X_val, Y_val


if __name__ == '__main__':
    process_and_load_data()
