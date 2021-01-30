import os
import sys
from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging

from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

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

# Warnings lead to an exception
np.seterr(all='raise')

# pseudo-GridsearchCV wrapper for custom grid search results
class PseudoGrid():
    def __init__(self, estimator, params):
        self.best_estimator_ = estimator
        self.best_params_ = params

    def predict(self, *args, **kwargs):
        return self.best_estimator_.predict(*args, **kwargs)

def create_model_template(network, shape, legacy):
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
                if not legacy:
                    model.add(Dropout(rate=0.5))
    
            if not legacy:
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


def model_cv(model_constr, modality, cognitive_data, feature, emb_type, cog_config, word_embedding, X, y, fold_size_lower_b):
    '''
    Performs grid search cross-validation using the given
    model construction function, configuration and
    training data.

    :param model_constr: model construction function
    :param config: embedding configuration dictionary
    :param X: training split of data (embeddings)    
    :param y: training split of labels (cognitive data)
    '''
    config = cog_config['wordEmbSpecifics'][word_embedding]
    param_grid = dict(layers=config["layers"], activation=config["activations"], input_dim=[X.shape[1]],
                      output_dim=[y.shape[1]], batch_size=config["batch_size"], epochs=config["epochs"])
    
    if emb_type == 'word':
        model = KerasRegressor(build_fn=model_constr, verbose=0)

        print("Applying standard grid search ...")
        grid = GridSearchCV(estimator=model,
                                      param_grid=param_grid,
                                      scoring='neg_mean_squared_error',
                                      cv=config['cv_split'])
        grid_result = grid.fit(X, y, verbose=0, validation_split=config["validation_split"])

        ss, pca, minmax = None, None, None

    elif emb_type == 'sentence':
        # Modify output dimensionality in case of EEG or fMRI (KPCA)
        if modality in ('eeg', 'fmri'):
            kpca_n_dims = cog_config.get('kpca_n_dims', None)
            if not kpca_n_dims:
                if modality == 'eeg':
                    kpca_n_dims = 32
                elif modality == 'fmri':
                    kpca_n_dims = 256
            kpca_gamma = cog_config.get('kpca_gamma', None)
            kpca_gamma = kpca_gamma if kpca_gamma else None
            kpca_kernel = cog_config.get('kpca_kernel', 'poly')
            
        # Can't do target transformation with GridsearchCV, thus manual implementation
        print("Applying custom grid search ...")

        kf = KFold(n_splits=config['cv_split'], random_state=None, shuffle=False)
        param_grid = list(ParameterGrid(param_grid))
        mean_scores = []

        for params in param_grid:
            cv_scores = []
            
            for idx, (train_index, test_index) in enumerate(kf.split(X)):
                # Need to create regressor anew EVERY time to avoid incremental fitting
                model = KerasRegressor(build_fn=model_constr, verbose=0)
                
                # Target transformers
                ss = StandardScaler() 
                minmax = MinMaxScaler()   

                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if modality in ('eeg', 'fmri'):
                    # KPCA dimensionality bounded by inner fold size (cap n_dims by data_size - 1)
                    kpca_inner_dims = min(kpca_n_dims, (y_train.shape[0] - 1))
                    if not idx:
                        print("Inner CV KernelPCA (n_dims: {} / kernel: {} / gamma: {})".format(kpca_inner_dims, kpca_kernel, kpca_gamma if kpca_gamma else 'sklearn default'))
                    pca = KernelPCA(kpca_inner_dims, kernel=kpca_kernel, gamma=kpca_gamma)
                    params['output_dim'] = kpca_inner_dims

                model.set_params(**params)

                # Target transformation within fold to prevent data leakage
                if modality in ('eeg', 'fmri'):
                    if cog_config.get('standardize', True):
                        if not 'standardize' in cog_config:
                            cprint('Missing "standardize" field, defaulting to True', 'yellow')
                        y_train = ss.fit_transform(y_train)
                        y_test = ss.transform(y_test)
                    y_train = pca.fit_transform(y_train)
                    y_test = pca.transform(y_test)
                y_train = minmax.fit_transform(y_train)
                y_test = minmax.transform(y_test)

                # Clip test values outside of [0, 1] (unseen by MinMaxScaler upon fitting)
                print('Test range before clipping:', np.min(y_test), np.max(y_test))
                y_test = np.clip(y_test, 0.0, 1.0)
                print('After clipping:', np.min(y_test), np.max(y_test))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                cv_scores.append(mse)
                
            mean_score = np.mean(cv_scores)
            mean_scores.append(mean_score)
            
        # Smallest value is best-performing
        best_id = np.argmin(mean_scores)
        best_params = param_grid[best_id]
        cprint("{} / {} / {} - Best score inner CV: {}".format(cognitive_data, feature, word_embedding, mean_scores[best_id]), "magenta")

        ss = StandardScaler()
        minmax = MinMaxScaler()
        if modality in ('eeg', 'fmri'):
            # KPCA dimensionality bounded by outer fold size (cap n_dims by data_size - 1)
            kpca_outer_dims = min(kpca_n_dims, (fold_size_lower_b - 1))
        # Retrain regressor on full data
        model = KerasRegressor(build_fn=model_constr, verbose=0)
        if modality in ('eeg', 'fmri'):
            # Reset output dim to correspond to outer fold size
            best_params['output_dim'] = kpca_outer_dims
        model.set_params(**best_params)
        if modality in ('eeg', 'fmri'):
            print("Outer CV KernelPCA (n_dims: {} / kernel: {} / gamma: {})".format(kpca_outer_dims, kpca_kernel, kpca_gamma if kpca_gamma else 'sklearn default'))
            pca = KernelPCA(kpca_outer_dims, kernel=kpca_kernel, gamma=kpca_gamma)

        # Target transform
        if modality in ('eeg', 'fmri'):
            if cog_config.get('standardize', True):
                y = ss.fit_transform(y)
            else:
                ss = None
            y = pca.fit_transform(y)
        else:
            ss = None
            pca = None
        y = minmax.fit_transform(y)

        # Wrap in pseudo-Grid object to keep logging code invariant
        grid_result = PseudoGrid(model.fit(X, y, verbose=0, validation_split=config["validation_split"]), best_params) # required for history
        grid = PseudoGrid(model, best_params)

    return grid, grid_result, ss, pca, minmax


def model_predict(grid, ss, pca, minmax, words, X_test, y_test, emb_type):
    '''
    Performs prediction for test data using given
    fitted GridSearchCV model.

    :param grid: fitted GridSearchCV model
    :param words: words corresponding to embedding
    :param X_test: test data (embeddings)
    :param y_test: test labels (cognitive data)
    '''
    y_pred = grid.predict(X_test)
    if y_test.shape[1] == 1:
        y_pred = y_pred.reshape(-1,1)
    
    # Apply transformations fitted on training targets to test targets
    if ss:
        y_test = ss.transform(y_test)
    if pca:
        y_test = pca.transform(y_test)
    if minmax:
        y_test = minmax.transform(y_test)

    # Clip test values outside of [0, 1] (unseen by MinMaxScaler upon fitting)
    if emb_type == 'sentence':
        print('Test range before clipping:', np.min(y_test), np.max(y_test))
        y_test = np.clip(y_test, 0.0, 1.0)
        print('After clipping:', np.min(y_test), np.max(y_test))
    error = np.abs(y_test - y_pred)

    word_error = np.hstack([words,error])
    
    if y_test.shape[1] ==1:
        mse = np.mean(np.square(error))
    else:
        mse = np.mean(np.square(error),axis=0)
    return mse, word_error


def model_loop(i, X_train, X_test, y_train, y_test, fold_size_lower_b, words_test, network, gpu_id, cog_config, modality, cognitive_data, feature, emb_type, word_embedding, legacy):
    '''
    Performs GridsearchCV on a single fold of the (outer) cross-validation and returns best model refitted on full data.
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if gpu_id is not None:
        gpu_count = 1
        soft_placement = True
    else:
        gpu_count = 0
        soft_placement = False
 
    tf_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                         inter_op_parallelism_threads=1,
                                         allow_soft_placement=soft_placement,
                                         device_count={'GPU': gpu_count, 'CPU': 1})

    if gpu_id is not None:
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        tf_config.gpu_options.visible_device_list = str(gpu_id)

    with tf.compat.v1.Session(config=tf_config) as session:
        init = tf.compat.v1.global_variables_initializer() # This reinitializes keras weights, so must be put before Keras loading
        set_session(session)
        session.run(init)
        with session.as_default():
            model_f = create_model_template(network, X_train[i].shape, legacy)
            grid, grid_result, ss, pca, minmax = model_cv(model_f,
                         modality,
                         cognitive_data,
                         feature,
                         emb_type,
                         cog_config,
                         word_embedding,
                         X_train[i],
                         y_train[i],
                         fold_size_lower_b)

        cprint("{} / {} / {} - Fold #{} GridsearchCV best params: {}".format(cognitive_data, feature, word_embedding, i + 1, \
        " | ".join(["{}: {}".format(k, v) for k, v in grid_result.best_params_.items()])),
        'magenta')
        
        mse, w_e = model_predict(grid,
                 ss, pca, minmax,
                 words_test[i],
                 X_test[i],
                 y_test[i],
                 emb_type)

    tf.compat.v1.reset_default_graph()
    clear_session()

    return grid, grid_result, mse, w_e


def model_handler(word_embedding,
                  modality,
                  cognitive_data,
                  feature,
                  cog_config,
                  emb_type,
                  words_test,
                  X_train,
                  y_train,
                  X_test,
                  y_test,
                  network,
                  gpu_id,
                  legacy):
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

    # Determine lower bound of fold size for outer KPCA (inner KPCA dim can fluctuate, as word/sentence-level error is not exported here)
    fold1_train_size = X_train[0].shape[0]
    fold1_test_size = X_test[0].shape[0]
    fold_size_lower_b = int(np.floor(((fold1_train_size + fold1_test_size)/len(X_train)) * len(X_train[1:])))

    if y_test[0].shape[1] == 1:
        word_error = np.array([emb_type, 'error'],dtype='str')
    else:
        # Word error dimensionality of multi-dim output depends on whether PCA is performed (eeg, fmri)
        if emb_type == 'sentence' and  modality in ('eeg', 'fmri'):
            word_error = np.array([emb_type] + ['e' + str(i) for i in range(cog_config['kpca_n_dim'])], dtype='str')
        else:
            word_error = np.array([emb_type] + ['e' + str(i) for i in range(1, y_test[0].shape[1]+1)], dtype='str')
    for i in range(len(X_train)):
        grid, grid_result, mse, w_e = model_loop(i, X_train, X_test, y_train, y_test, fold_size_lower_b, words_test, network, gpu_id, cog_config, modality, cognitive_data, feature, emb_type, word_embedding, legacy)

        grids_result.append(grid_result)
        
        mserrors.append(mse)

        # Resize header if necessary
        if not i:
            word_error = word_error[:w_e.shape[1]]
        try:
            word_error = np.vstack([word_error,w_e])
        except:
            print(i, w_e)

    return word_error, grids_result, mserrors
