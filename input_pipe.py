"""
INPUT PIPELINE.

This pipeline is organized as a class that can be called from a Jupyter Notebook
"""
import pickle
import numpy as np
import pandas as pd

from dataprep_tools import *



# TODO: in load_dataframe() assicurati l'ordinamento per colonne





def load_dataframe(path_to_data):

    df = pd.read_csv(path_to_data + 'train_2.csv')
        
    # drop rows that are all NaN's
    df.dropna(axis = 'rows', how = 'all', inplace = True)
    return df


def get_time_schema():
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
    Attaches page data to main df:
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

    # Loads raw data and main model hyperparams
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







#################################################################################################







class DataProcessor():
    """
    Main wrapper for the whole pipe. This object is to be instantiated for loading
    and preprocessing the dataset cleanly from Jupyter Notebook or other scripts.
    """

    def __init__(self, path_to_data):
        import time
        import pickle
        import numpy as np
        import pandas as pd
        from dataprep_tools import *   # Homemade module

        self.path_to_data = path_to_data


    def __load_data(path_to_data):
        df1 = pd.read_csv(path_to_data + 'train_1.csv')
        df2 = pd.read_csv(path_to_data + 'train_2.csv')
        df = df1.merge(df2, on = 'Page')

        # TODO: Assicurati l'ordinamento per colonne
        params = open("main_hyperparams.pkl","wb")
        return df, params


    def __impute_nan(df):

        # Left-zero padding

        # Taking care of the ones in the middle

        return df


    def __scale_trends(df):
        return df, scale_params


    def __get_time_schema(df):
        """ Returns vectors with patterns for time-related variables (year/week days)
        in [0,1] range, to be repeated on all trends. """
        daterange = pd.date_range(df.columns[1], df.columns[-1], freq='D').to_series()

        weekdays = daterange.dt.dayofweek
        weekdays = weekdays.values / weekdays.max()
        yeardays = daterange.dt.dayofyear
        yeardays = yeardays.values / yeardays.max()
        return weekdays, yeardays

    def __attach_page_data(df):
        """
        Attaches page data to main df:
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


    def process_and_load_data(self):
        start = time.time()

        print('Loading raw data and main hyperparams.')
        df, params = self.__load_data(self.path_to_data)

        print('Imputation of missing data.')
        df = self.__impute_nan(df)

        print('Scaling of trends by language group.')
        df, scaling_params = self.__scale_trends(df, self.path_to_data)

        weekdays, yeardays = self.__get_time_schema(df)

        df = self.__attach_page_data(df)

        X_final, Y_final = self.__get_train_and_target_data(df, len_test_series = params['len_test_series'])

        print('\nPreprocessing and loading completed in {}ss'.format(round(time.time()-start, 2)))

        return X_final, Y_final
#
