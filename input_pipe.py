"""
INPUT PIPELINE.

This pipeline is organized as a class that can be called from a Jupyter Notebook
"""
import pickle
import numpy as np
import pandas as pd

from dataprep_tools import *



# TODO: assicurati che in load_dataframe() avvenga l'ordinamento per colonne





def load_dataframe(path_to_data):

    df = pd.read_csv(path_to_data + 'train_2.csv')

    # drop rows that are all NaN's
    df.dropna(axis = 'rows', how = 'all', inplace = True)
    return df


def get_time_schema(df):
    """ Returns vectors with patterns for time-related variables (year/week days)
    in [0,1] range, to be repeated on all trends. """
    daterange = pd.date_range(df.columns[1], df.columns[-1], freq='D').to_series()

    weekdays = daterange.dt.dayofweek
    weekdays = weekdays.values / weekdays.max()
    yeardays = daterange.dt.dayofyear
    yeardays = yeardays.values / yeardays.max()
    return weekdays, yeardays


def attach_page_data(df):
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


def process_and_load_data(path_to_data):
    """
    Main wrapper for the whole pipe. This object is to be instantiated for loading
    and preprocessing the dataset cleanly from Jupyter Notebook or other scripts.
    """

    print('Loading raw data and main hyperparams.')
    df = load_dataframe(path_to_data)
    params = open("main_hyperparams.pkl","wb")

    print('Imputation of missing data.')
    df = impute_nan(df)

    print('Scaling of trends by language group.')
    df, scaling_params = scale_trends(df, path_to_data)

    weekdays, yeardays = get_time_schema(df)

    df = attach_page_data(df)

    # Prepares data matrices ready for Jupyter Notebooks
    X_final, Y_final = get_train_and_target_data(df, len_test_series = params['len_test_series'])

    return X_final, Y_final
