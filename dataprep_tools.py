"""
PRE-PROCESSING TOOLS FOR INPUT PIPELINE.
"""
import pickle
import re
import numpy as np
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


def impute_nan(x):
    """
    Applies to forms of missing values imputation.
    1. left NaN's mean the series didn't start and are imputed with zero
    2. Other NaN's within the series are imputed with a ML model for imputations
    """
    import numpy as np
    
    # Fill left NaN's with 0
    if np.isnan(x[0]):
        cumsum = np.cumsum(np.isnan(x))    
        x[ :np.argmax(cumsum[:-1]==cumsum[1:]) +1] = 0
    
    # Impute internal NaN's with imputation model
    if np.sum(np.isnan(x)) > 0:
        x_imputed = model.predict(x)
        return x_imputed
    else:
        return x


def scale_trends(df, path_to_data):
    """
    Scales trends divided by language groups. It splits df into sub-df's by
    language, and applies a robust custom scaling in two steps:
        1. log( x + 1 )
        2. Robust min-max scaling to 0 and 99th percentile (by language group)
    All 99th percentiles are saved in scaling_dict that is then pickled in
    .../data/ folder, to allow for repeated applications of the model.
    """
    import pickle
    import numpy as np
    import pandas as pd

    scaling_dict = {}

    for language in languages:
        sdf = df[ df['language'] == language ]
        page = sdf['Page']
        X = sdf.drop('Page', axis = 1).values

        # log(x+1) and robust scale to [0, 99th percentile]
        X = np.log(X + 1)
        percentile_99th = np.percentile(X.reshape((X.size, )), 99)
        X = ( X - np.min(X) ) / ( percentile_99th - np.min(X) )

        # store 99th percentile into scaling_dict
        scaling_dict[language] = percentile_99th

    # pickle this dict into save_path
    d = open(path_to_params + 'scaling_dict.pkl','wb')
    pickle.dump(scaling_dict, f)
    d.close()

    return df


def RNN_dataprep(trend, len_input):
    """
    Main processing function.
    From each trend, returns 3D np.array defined by:
        ( no obs. , length input series , no input variables )
    Steps:
        1. From trend, trend vars and page vars, returns 2D np.array with one col
            per variable
        2. Creates empty 3D np.array, and fills it variable by variable by running
            _univariate_processing(). It takes a variable, turns it into a 2D
            matrix, then pastes into the final 3D matrix as a slice
    """
    import numpy as np

    def _univariate_processing(s, window, stepsize=1):
        '''Process single vars, gets a 'sliding window' 2D array out of a 1D var'''
        nrows = ((s.size-window)//stepsize)+1
        n = s.strides[0]
        return np.lib.stride_tricks.as_strided(s, shape=(nrows,window), strides=(stepsize*n,n))
    
    page_vars = trend[ -16: ]  # last 16 cols are page data
    
    x = trend[ 365:-16 ]   # actual time series
    # trend = trend[ 365:-16 ]
    
    # Missing values imputation    
    

    trend_lag_year = trend[ 0:len(x) ]   # 365 days ahead
    trend_lag_quarter =  = trend[ 90:len(trend)+90 ]  # 90 days ahead
    
    # scalar page vars (language, website, access, agent) are repeated for whole col
    page_vars = np.column_stack([ np.repeat(j, len(trend)) for j in page_vars ])
    
    # Make a 2D matrix for each trend, composed of:
    T = np.column_stack([
        x,
        trend_lag_year,     # year time lag
        trend_lag_quarter,  # quarter time lag
        page_vars,          # page vars
        weekdays, yeardays  # temporal position in week and year (scaled to [0, 1])
    ])

    # Use data from T to fill preprocessed matrix for RNN
    X = np.empty((T.shape[0]-len_input+1, len_input, T.shape[1]))

    for i in range(T.shape[1]):
        X[ : , : , i ] = _univariate_processing(T[:,i], len_input)

    X = X.astype(np.float32)
    return X
