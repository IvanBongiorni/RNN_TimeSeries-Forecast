"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-08-17

Calls training functions
"""
import tensorflow as tf
from pdb import set_trace as BP
# local imports
import tools, model


def get_processed_batch(batch, params):
    '''
    Once an observation (time series) has been loaded, processes it for RNN inputs,
    making it a 3D array with shape:
        ( n. obs ; input lenght ; n. variables )

    Batch is then cut into input (multivariate) and target (univariate) sets, x and y.
    This function is called both for Train and Validation steps.
    '''
    import numpy as np
    import tools  # local import

    batch = tools.RNN_multivariate_processing(
        array = batch,
        len_input = params['len_input'] + params['len_prediction'] # Sum them to get X and Y data
    )

    # Sample a mini-batch from it
    sample = np.random.choice(batch.shape[0], size=np.min([batch.shape[0], params['batch_size']]), replace = False)
    batch = batch[ sample , : , : ]

    y = batch[ : , params['len_input']: , 0 ]  # target trend (just univariate)
    x = batch[ : , :params['len_input'] , : ]
    return x, y


def train(model, params):
    '''
    ## TODO:  [ doc to be rewritten ]
    '''
    import os
    import pickle
    import time
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    @tf.function
    def train_on_batch(X_batch, Y_batch):
        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(tf.math.abs(model(X_batch) - Y_batch))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    # Get list of all Training and Validation observations
    X_files = os.listdir( os.getcwd() + '/data_processed/Train/' )
    if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
    if '.gitignore' in X_files: X_files.remove('.gitignore')
    X_files = np.array(X_files)

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling filename index
        if params['shuffle']:
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace=False) ]

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # Load raw series, keep Train data, get input and target batch
            batch = np.load('{}/data_processed/Train/{}'.format(os.getcwd(), X_files[iteration]), allow_pickle=True)
            batch = batch[ :-int(len(batch)*params['val_size']) , : ]
            X_batch, Y_batch = get_processed_batch(batch, params)

            # BP()

            # Train model
            train_loss = train_on_batch(X_batch, Y_batch)

            # Once in a while check and print progress on Validation data
            if iteration % 100 == 0:
                # Repeat loading but keep Validation data this time
                batch = np.load('{}/data_processed/Train/{}'.format(os.getcwd(), X_files[iteration]), allow_pickle=True)
                # batch = batch[ : , :-(params['len_input']+params['len_prediction']) , : ]
                batch = batch[ :-(int(len(batch)*params['val_size'])+params['len_input']) , : ]
                X_batch, Y_batch = get_processed_batch(batch, params)

                validation_loss = tf.reduce_mean(tf.math.abs(model(X_batch) - Y_batch))

                print('{}.{}   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
                    epoch, iteration, train_loss, validation_loss, round(time.time()-start, 4)))

    print('\nTraining complete.\n')

    model.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Model saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    return None


def main():
    ''' Wrapper of training pipeline. '''
    import os
    import yaml
    import tensorflow as tf
    # local imports
    import model
    import tools

    print('\nStart training pipeline.')

    print('\n\tLoading configuration parameters.')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    if params['use_gpu']:
        print('\tSetting GPU configurations.')
        tools.set_gpu_configurations(params)

    # Check if pretrained model with 'model_name' exists, otherwise create a new one
    if params['model_name']+'.h5' in os.listdir(os.getcwd()+'/saved_models/'):
        print('Loading existing model: {}.'.format(params['model_name']))
        regressor = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    else:
        print('\nNew model created as: {}\n'.format(params['model_name']))
        regressor = model.build(params)

    regressor.summary()

    print('\nStart training.\n')
    # Create/load model and train
    train(regressor, params)


    return None


if __name__ == '__main__':
    main()
