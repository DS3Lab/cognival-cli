import os
import sys
from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging

from sklearn.model_selection import GridSearchCV
import numpy as np

from termcolor import cprint
import importlib

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session, clear_session
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Reshape

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.compat.v1.keras.wrappers.scikit_learn import KerasRegressor
sys.stderr = stderr

def create_model_template(network, shape):
    def create_model(layers, activation, input_dim, output_dim):
        '''
        Builds and compiles a Keras Sequential model based on the given
        parameters.

        :param layers: [hiddenlayer1_nodes,hiddenlayer2_nodes,...]
        :param activation: e.g. relu
        :param input_dim: number of input nodes
        :return: Keras model
        '''
        model = Sequential()
        if network == 'mlp':
            for i, nodes in enumerate(layers):
                if i==0:
                    model.add(Dense(nodes,input_dim=input_dim, activation=activation))
                else:
                    model.add(Dense(nodes, activation=activation))
                model.add(Dropout(rate=0.5))
                
            model.add(BatchNormalization())

        elif network == 'cnn':
            if len(layers) < 2:
                raise RuntimeError("CNN must have at least 2 layers (1 Conv1D layer and 1 dense layer)")
            for i, nodes in enumerate(layers[:-1]):
                if i == 0:
                    model.add(Reshape((shape[1], 1), input_shape=(shape[1], )))
                    model.add(Conv1D(nodes,
                                  input_shape=shape[1:],
                                  kernel_size=3,
                                  padding='valid',
                                  activation=activation,
                                  strides=2))
                else:
                    model.add(Conv1D(nodes,
                                  kernel_size=3,
                                  padding='valid',
                                  activation=activation,
                                  strides=1))
                model.add(MaxPooling1D()) 
            model.add(BatchNormalization())
            # Last hidden layer is normal dense layer
            model.add(Flatten())
            model.add(Dense(layers[-1], activation=activation))
        else:
            raise ValueError("Network must either be 'mlp' or 'cnn'")
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse',optimizer='adam')
        return model

    return create_model


def model_cv(model_constr, config, X_train, y_train):
    '''
    Performs grid search cross-validation using the given
    model construction function, configuration and
    training data.

    :param model_constr: model construction function
    :param config: embedding configuration dictionary
    :param X_train: training data (embeddings)    
    :param y_train: training labels (cognitive data)
    '''
    model = KerasRegressor(build_fn=model_constr, verbose=0)

    param_grid = dict(layers=config["layers"], activation=config["activations"], input_dim=[X_train.shape[1]],
                      output_dim=[y_train.shape[1]], batch_size=config["batch_size"], epochs=config["epochs"])
    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error', cv=config['cv_split'])
    grid_result = grid.fit(X_train,y_train, verbose=0, validation_split=config["validation_split"])

    return grid, grid_result


def model_predict(grid, words, X_test, y_test):
    '''
    Performs prediction for test data using given
    fitted GridSearchCV model.

    :param grid: fitted GridSearchCV model
    :param words: words corresponding to embedding
    :param X_test: test data (embeddings)
    :param y_test: test labels (cognitive data)
    '''
    y_pred = grid.predict(X_test)
    if y_test.shape[1] ==1:
        y_pred = y_pred.reshape(-1,1)
    error = y_test - y_pred
    word_error = np.hstack([words,error])
    if y_test.shape[1] ==1:
        mse = np.mean(np.square(error))
    else:
        mse = np.mean(np.square(error),axis=0)
    return mse, word_error


def model_loop(i, X_train, X_test, y_train, y_test, words_test, network, gpu_id, config, cognitive_data, feature, word_embedding):
    '''
    Performs GridsearchCV on a single fold of the (outer) cross-validation and returns best model refitted on full data.
    '''
    if gpu_id is not None:
        gpu_count = 1
        soft_placement = True
    else:
        gpu_count = 0
        soft_placement = False
 
    tf_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                         inter_op_parallelism_threads=1,
                                         allow_soft_placement=False,
                                         device_count={'GPU': soft_placement, 'CPU': 1})

    if gpu_id is not None:
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25
        tf_config.gpu_options.visible_device_list = str(gpu_id)

    
    with tf.compat.v1.Session(config=tf_config) as session:
        init = tf.compat.v1.global_variables_initializer() # This reinitializes keras weights, so must be put before Keras loading
        set_session(session)
        session.run(init)
        with session.as_default():
            grid, grid_result = model_cv(create_model_template(network, X_train[i].shape),
                         config,
                         X_train[i],
                         y_train[i])

        cprint("{} / {} / {} - Fold #{} GridsearchCV best params: {}".format(cognitive_data, feature, word_embedding, i + 1, \
        " | ".join(["{}: {}".format(k, v) for k, v in grid_result.best_params_.items()])),
        'magenta')
        
        mse, w_e = model_predict(grid,
                 words_test[i],
                 X_test[i],
                 y_test[i])

    tf.compat.v1.reset_default_graph()
    clear_session()

    return grid, grid_result, mse, w_e


def model_handler(word_embedding,
                  cognitive_data,
                  feature,
                  config,
                  emb_type,
                  words_test,
                  X_train,
                  y_train,
                  X_test,
                  y_test,
                  network,
                  gpu_id):
    '''
    Performs cross-validation on chunks of training data to determine best parametrization
    based on parameter grid given in config. Predicts with best-performing model on chunks
    of test data. Returns word error, best-performing model and MSEs.

    :param config: embedding configuration dictionary
    :param emb_type: 'word' or 'sentence' embeddings
    :param words_test: words corresponding to test data
    :param X_train: training data (embeddings)    
    :param y_train: training labels (cognitive data)
    :param X_test: test data (embeddings)
    :param y_test: test labels (cognitive data)
    '''
    grids = []
    grids_result = []
    mserrors = []

    if y_test[0].shape[1] == 1:
        word_error = np.array([emb_type, 'error'],dtype='str')
    else:
        word_error = np.array([emb_type] + ['e' + str(i) for i in range(1, y_test[0].shape[1]+1)], dtype='str')

    for i in range(len(X_train)):
        grid, grid_result, mse, w_e = model_loop(i, X_train, X_test, y_train, y_test, words_test, network, gpu_id, config, cognitive_data, feature, word_embedding)

        grids.append(grid)
        grids_result.append(grid_result)
        
        mserrors.append(mse)
        word_error = np.vstack([word_error,w_e])

    return word_error, grids_result, mserrors
