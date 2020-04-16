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
    test_cutoff = int()
    df = df.iloc[  ]

    return df, validation, test


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


def get_train_and_target_data(df, len_input, len_test):
    languages = ['en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na']
    X_final = []

    for language in languages:
        print('Preprocessing of language group: {}'.format(language))
        language_pick = 'language_{}'.format(language)
        sdf = df[ df[language_pick] == 1 ].values

        sdf = scale_data(sdf)

        # Iterate processing on each trend
        sdf = [ RNN_dataprep(sdf[i,:,:], len_input) for i in range(sdf.shape[0]) ]
        sdf = np.concatenate(sdf, axis = 0)

        X_final.append(sdf)

    X_final = np.concatenate(X_final, axis = 0)

    X_final = X_final[ : , :-len_test , : ]
    Y_final = Y_final[ : , -len_test: , : ]
    Y_final = np.squeeze(Y_final[ : , : , 0 ])

    return X_final, Y_final





################################################################################





def process_and_load_data(path_to_data):
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
    df, page_data = load_dataframe(current_path + '/data/train2.csv')

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

    # Save scaling params once it's done

    # imputation on final matrix from model - trained on 3D input

    # split in T-V-T   ***  HERE  ***


    # df = attach_page_data(df)
    #
    # # Prepares data matrices ready for Jupyter Notebooks
    # X_final, Y_final = get_train_and_target_data(df, len_test_series = params['len_test_series'])


    ### TODO: DOBBIAMO RESTITUIRE ARRAY SIA TRAIN CHE VALIDATION

    return X
