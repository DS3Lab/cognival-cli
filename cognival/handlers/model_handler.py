import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.compat.v1.keras.activations import relu, linear
from tensorflow.compat.v1.keras.wrappers.scikit_learn import KerasRegressor
sys.stderr = stderr
from sklearn.model_selection import GridSearchCV
import numpy as np

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
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=input_dim, activation=activation))
        else:
            model.add(Dense(nodes, activation=activation))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse',optimizer='adam')

    return model


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


def model_handler(config, words_test, X_train, y_train, X_test, y_test):
    '''
    Performs cross-validation on chunks of training data to determine best parametrization
    based on parameter grid given in config. Predicts with best-performing model on chunks
    of test data. Returns word error, best-performing model and MSEs.

    :param config: embedding configuration dictionary
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
        word_error = np.array(['word', 'error'],dtype='str')
    else:
        word_error = np.array(['word'] + ['e' + str(i) for i in range(1,y_test[0].shape[1]+1)], dtype='str')
    for i in range(len(X_train)):
        grid, grid_result = model_cv(create_model,config,X_train[i],y_train[i])
        grids.append(grid)
        grids_result.append(grid_result)
        mse, w_e = model_predict(grid,words_test[i],X_test[i],y_test[i])
        mserrors.append(mse)
        word_error = np.vstack([word_error,w_e])
    return word_error, grids_result, mserrors
