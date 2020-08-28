"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-08-17

Calls training functions
"""
import tensorflow as tf

# local modules
import tools, model


def process_series(batch, params):
    import numpy as np
    import deterioration, tools  # local imports

    X_batch = tools.RNN_multivariate_processing(
        array = batch, 
        len_input = params['len_input']+params['len_pred'] # Sum them to get X and Y data
    )

    sample = np.random.choice(X_batch.shape[0], size=np.min([X_batch.shape[0], params['batch_size']]), replace = False)
    
    Y_batch = X_batch[ : , params['len_input']: , 0 ]  # Extract target trend (just univariate)
    Y_batch = np.expand_dims(Y_batch, axis=-1)

    X_batch = X_batch[ : , :params['len_input'] , 1: ]
    
    return X_batch, Y_batch, mask


def train(model, params):
    '''
    ## TODO:  [ doc to be rewritten ]
    '''
    import time
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def train_on_batch(X_batch, Y_batch):
        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(tf.math.abs(model(X_batch) - Y_batch))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    # Get list of all Training and Validation observations
    X_files = os.listdir( os.getcwd() + '/data_processed/Training/' )
    if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
    if '.gitignore' in X_files: X_files.remove('.gitignore')
    X_files = np.array(X_files)

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling row index
        if params['shuffle']:
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace=False) ]

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            X_batch, Y_batch = process_series(batch, params)

            current_loss = train_on_batch(X_batch, Y_batch, mask)

            # Save and print progress each 50 training steps
            if iteration % 100 == 0:
                v_file = np.random.choice(V_files)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                X_batch, Y_batch = process_series(batch, params)

                mask = np.expand_dims(mask, axis=-1)
                validation_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(model(X_batch), mask) - tf.math.multiply(Y_batch, mask)))

                print('{}.{}   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
                    epoch, iteration, current_loss, validation_loss, round(time.time()-start, 4)))

    print('\nTraining complete.\n')

    model.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Model saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    return None


def main():
    # Instantiate model

    # 

    
    return None


if __name__ == '__main__':
    main()
