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
    df.drop('Page', axis = 1, inplace = True)
    return df, page_data


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
    df, page_data = process_page_data(page_data)

    print('Creating time schema')
    weekdays, yeardays = get_time_schema(df)

    print('Loading imputation model: {}'.format(params['imputation_model']))
    model = tf.keras.models.load_model(current_path + '/saved_models/' + params['imputation_model'])

    X_train = []
    scaling_dict = {}

    print('Preprocessing trends by language group.')
    X_train = []
    X_test = []

    scaling_dict = {}   # This is to save scaling params - by language subgroup

    for language in languages:
        start = time.time()

        sdf = df[ page_data['language'] == language ].values
        sdf_page_data = page_data[ page_data['language'] == language ].values

        # Fill left-NaN's with zero
        for i in range(sdf.shape[0]):
            sdf[ i , : ] = series
            series = tools.left_zero_fill( series )
            series = tools.right_trim_nan( series )
            sdf[ i , : ] = series

        ###
        ### SPLIT IN TRAIN - VAL - TEST
        ###   This must be done differently from imputation project
        # sdf_train =
        # sdf_val =
        # sdf_test =

        ### IMPORTANT: PAGE DATA TOO MUST BE FILTERED THE SAME WAY
        # sdf_page_data_train
        # sdf_page_data_val
        # sdf_page_data_test

        # Scale and save param into dict
        scaling_percentile = np.percentile(sdf_train, 99)
        sdf_train = scale_trends(sdf, scaling_percentile)
        sdf_val = scale_trends(sdf_val, scaling_percentile)
        sdf_test = scale_trends(sdf_val, scaling_percentile)
        scaling_dict[language] = scaling_percentile

        # Process to RNN format ('sliding window' to input series) and pack into final array
        sdf_train = [ tools.RNN_dataprep(               ) for series in sdf_train ]
        sdf_train = np.concatenate(                  )

        sdf_val = [ tools.RNN_dataprep(series, params) for series in sdf_val ]
        sdf_val = np.concatenate(sdf_val)

        X_train.append(sdf_train)
        X_val.append(sdf_val)

        print("\tSub-dataframe for language '{}' executed in {} ss.".format(language, round(time.time()-start, 2)))

    # Concatenate datasets (shuffle X_train for batch training)
    X_train = np.concatenate(X_train)
    shuffle = np.random.choice(X.shape[0], X.shape[0] replace = False)
    X_train = X_train[ shuffle , : ]
    X_val = np.concatenate(X_val)

    ##  Imputation
    print('Loading imputation model: {}.h5'.format(params['imputation_model']))
    imputer = tf.keras.load_model('{}/imputation_model/{}.h5'.format(current_path, params['imputation_model']))
    # Order of variables is:
    # t,
    # trend_lag_quarter,
    # trend_lag_year,
    # and then: weekdays,  yeardays  + four page variables

    # Imputation must happen for the first three vars (trend data)
    # line by line to avoid excessive computational costs
    for var in range(3):
        for i in range():
            X_train[ i , : , var ] = imputer.predict( X_train[ i , : , var ] )
            X_test[ i , : , var ] = imputer.predict( X_test[ i , : , var ] )


    #### IMPORTANT:
    #### [ NUMPY 2 PANDAS 2 PICKLE ] DOESN'T WORK WITH 3D ARRAYS
    # Must use pickle library

    # Then pickle all, ready for training
    pickle.dump(X_train, open( os.getcwd() + '/data_processed/X_train.pkl' ))
    pickle.dump(X_test, open(os.getcwd() + '/data_processed/X_val.pkl'))

    # Save scaling params to file
    yaml.dump(scaling_dict,
              open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'),
              default_flow_style = False)

    return None


if __name__ == '__main__':
    process_and_load_data()
