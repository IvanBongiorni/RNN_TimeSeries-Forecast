"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-04-10

PRE-PROCESSING TOOLS FOR INPUT PIPELINE.
"""

import pickle
import re
import numpy as np
import numba
import pandas as pd


def process_url(url):
    """
    Extracts four variables from URL string:
        language:  code - with 'na' for 'no language detected'
        website:   what type of website: 'wikipedia', 'wikimedia', 'mediawiki'
        access:    type of access (e.g.: mobile, desktop, both, ...)
        agent:     type of agent
    """
    import re
    import numpy as np
    import pandas as pd

    if '_en.' in url: language = 'en'
    elif '_ja.' in url: language = 'ja'
    elif '_de.' in url: language = 'de'
    elif '_fr.' in url: language = 'fr'
    elif '_zh.' in url: language = 'zh'
    elif '_ru.' in url: language = 'ru'
    elif '_es.' in url: language = 'es'
    else: language = 'na'

    if 'wikipedia' in url: website = 'wikipedia'
    elif 'wikimedia' in url: website = 'wikimedia'
    elif 'mediawiki' in url: website = 'mediawiki'

    access, agent = re.split('_', url)[-2:]

    url_features = pd.DataFrame({
        'language': [language],
        'website': [website],
        'access': [access],
        'agent': [agent]
    })
    return url_features


@numba.jit(python = True)
def left_fill_nan(x):
    """ Fills all left NaN's with zeros, leaving others intact. """
    import numpy as np

    if np.isnan(x[0]):
        cumsum = np.cumsum(np.isnan(x))
        x[ :np.argmax(cumsum[:-1]==cumsum[1:]) +1] = 0
    return x


@numba.jit(python = True)
def scale_trends(X, params, language):
    """
    Takes a linguistic sub-dataframe and applies a robust custom scaling in two steps:
        1. log( x + 1 )
        2. Robust min-max scaling to [ 0, 99th percentile ]

    Returns scaled sub-df and scaling percentile, to be saved later in scaling dict
    """
    import numpy as np

    ### TODO: IMPORTANTE: La scalatura deve avvenire sui dati di Training,
    #   escludendo quelli di Validation

    # Scaling parameters must be calculated without Validation data
    percentile_99th = np.percentile(X.reshape((X.size, )), 99)


    # log(x+1) and robust scale to [0, 99th percentile]
    X = np.log(X + 1)
    X = ( X - np.min(X) ) / ( percentile_99th - np.min(X) )

    return df, percentile_99th


@numba.jit(python = True)
def right_trim_nan(x):
    """ Trims all NaN's on the right """
    import numpy as np

    if np.isnan(x[-1]):
        cut = np.argmax(np.isfinite(x[::-1]))
        return x[ :-cut ]
    else:
        return x


@numba.jit(python = True)
def univariate_processing(variable, window):
    '''
    Process single vars, gets a 'sliding window' 2D array out of a 1D var
    TO be iterated for each variable in RNN_dataprep().
    '''
    import numpy as np
    V = np.empty((len(variable)-window+1, window))  # 2D matrix from variable
    for i in range(V.shape[0]):
        V[i,:] = variable[i : i+window]
    return V.astype(np.float32)


@numba.jit(python = True)
def RNN_dataprep(t, page_vars, day_week, day_year, params):
    """
    Processes a single trend, to be iterated.
    From each trend, returns 3D np.array defined by:
        ( no obs. , length input series , no input variables )
    Where variables, stored in page object, are:
        - trend
        - quarter (-180) and year (-365) lags
        - one-hot page variables: language, website, access, agent
        - day of the week and day of the year in [0, 1]

    Apply right trim. If the resulting trend is too short, discard the observation.
    If it's long enough (len train + len prediction)

    Steps:
        1. From trend, trend vars and page vars, returns 2D np.array with one col
            per variable
        2. Creates empty 3D np.array, and fills it variable by variable by running
            _univariate_processing(). It takes a variable, turns it into a 2D
            matrix, then pastes into the final 3D matrix as a slice
    """
    import numpy as np

    # Trim trend to right length
    t = right_trim_nan(t)

    if len(t) < 365 + params['len_input'] + params['len_prediction']:
        return None

    # Cut trend and time lags
    trend_lag_year = t[ :-365 ]
    trend_lag_quarter = t[ 180:-180 ]
    t = t[ 365: ]

    # Add weekday and year day information
    day_week = day_week[ :len(t) ]
    day_year = day_year[ :len(t) ]

    # Make a 2D matrix of trend data
    T = np.column_stack([
        t,
        trend_lag_quarter,
        trend_lag_year,
        weekdays,
        yeardays
    ])

    # Attach page variables in the same format
    page_vars = np.repeat(p, len(t)).reshape((len(page_vars), len(t))).T #make 2D
    T = np.hstack([ T, page_vars ])

    # Apply actual RNN preprocessing

    # X_processed = np.empty((T.shape[0]-params['len_input']+1, params['len_input'], T.shape[1]))

    # X_processed = []
    # for i in range(T.shape[1]):
    #     X_processed.append( univariate_processing(T[:,i], len_input) )

    X_processed = [ univariate_processing(T[:,i], len_input) for i in range(T.shape[1]) ]
    X_processed = np.concatenate(X_processed)

    return X_processed
