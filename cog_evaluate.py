import argparse
from datetime import datetime

#own modules
from handlers.data_handler import data_handler
from handlers.model_handler import model_handler
from handlers.file_handler import *

def handler(config, word_embedding, cognitive_data, feature):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, applies a model (as per configuration) and returns
    per-word errors, resulting grids and mean squared error (MSE).
    Wraps data_handler and model_handler.

    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    '''

    words_test, X_train, y_train, X_test, y_test = data_handler(config,word_embedding,cognitive_data,feature)
    word_error, grids_result, mserrors = model_handler(config["cogDataConfig"][cognitive_data]["wordEmbSpecifics"][word_embedding],
                                         words_test, X_train, y_train, X_test, y_test)

    return word_error, grids_result, mserrors


def run(config, word_embedding, cognitive_data, feature):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, runs model, logs results and prepares output for plotting.

    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    '''

    ##############################################################################
    #   Create logging information
    ##############################################################################

    logging = {"folds":[]}

    logging["wordEmbedding"] = word_embedding
    logging["cognitiveData"] = cognitive_data
    logging["feature"] = feature

    ##############################################################################
    #   Run model
    ##############################################################################

    startTime = datetime.now()

    word_error, grids_result, mserrors = handler(config, word_embedding, cognitive_data, feature)

    history = {'loss':[],'val_loss':[]}
    loss_list =[]
    val_loss_list =[]

    ##############################################################################
    #   logging results
    ##############################################################################

    for i in range(len(grids_result)):
        fold = {}
        logging['folds'].append(fold)
        # BEST PARAMS APPENDING
        for key in grids_result[i].best_params_:
            logging['folds'][i][key.upper()] = grids_result[i].best_params_[key]
        if config['cogDataConfig'][cognitive_data]['type'] == "multivariate_output":
            logging['folds'][i]['MSE_PREDICTION_ALL_DIM:'] = list(mserrors[i])
            logging['folds'][i]['MSE_PREDICTION:'] = np.mean(mserrors[i])
        elif config['cogDataConfig'][cognitive_data]['type'] == "single_output":
            logging['folds'][i]['MSE_PREDICTION:'] = mserrors[i]

        logging['folds'][i]['LOSS: '] = grids_result[i].best_estimator_.model.history.history['loss']
        logging['folds'][i]['VALIDATION_LOSS: '] = grids_result[i].best_estimator_.model.history.history['val_loss']

        loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['loss'],dtype='float'))
        val_loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['val_loss'], dtype='float'))

    if config['cogDataConfig'][cognitive_data]['type'] == "multivariate_output":
        mserrors = np.array(mserrors, dtype='float')
        mse = np.mean(mserrors, axis=0)
        logging['AVERAGE_MSE_ALL_DIM'] = list(mse)
        logging['AVERAGE_MSE']= np.mean(mse)
    elif config['cogDataConfig'][cognitive_data]['type'] == "single_output":
        mse = np.array(mserrors, dtype='float').mean()
        logging['AVERAGE_MSE'] = mse


    ##############################################################################
    #   Prepare results for plot
    ##############################################################################

    history['loss'] = np.mean([loss_list[i] for i in range (len(loss_list))],axis=0)
    history['val_loss'] = np.mean([val_loss_list[i] for i in range(len(val_loss_list))], axis=0)

    timeTaken = datetime.now() - startTime
    logging["timeTaken"] = str(timeTaken)

    return logging, word_error, history

def main():
    '''
    CLI argument parsing and input sanity checking, execution and exporting
    results.
    '''

    ##############################################################################
    #   Set up of command line arguments to run the script
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path and name of configuration file",
                        nargs='?', default='config/setupConfig.json')
    parser.add_argument("-c", "--cognitive_data", type=str, default=None,
                        help="cognitive data to train the model")
    parser.add_argument("-f", "--feature", type=str,
                        default=None, help="feature of the dataset to train the model")
    parser.add_argument("-w", "--word_embedding", type=str, default=None,
                        help="word embedding to train the model")

    args = parser.parse_args()

    configFile = args.config_file
    cognitive_data = args.cognitive_data
    feature = args.feature
    word_embedding = args.word_embedding

    config = update_version(configFile)

    ##############################################################################
    #   Check for correct data inputs
    ##############################################################################

    while (word_embedding not in config['wordEmbConfig']):
        word_embedding = input("ERROR Please enter correct wordEmbedding:\n")
        if word_embedding == "x":
            exit(0)

    while (cognitive_data not in config['cogDataConfig']):
        cognitive_data = input("ERROR Please enter correct cognitive dataset:\n")
        if cognitive_data == "x":
            exit(0)

    if config['cogDataConfig'][cognitive_data]['type'] == "single_output":
        while feature not in config['cogDataConfig'][cognitive_data]['features']:
            feature = input("ERROR Please enter correct feature for specified cognitive dataset:\n")
            if feature == "x":
                exit(0)
    else:
        feature = "ALL_DIM"

    startTime = datetime.now()

    logging, word_error, history = run(config, word_embedding, cognitive_data, feature)

    ##############################################################################
    #   Saving results
    ##############################################################################

    write_results(config, logging, word_error, history)

    timeTaken = datetime.now() - startTime
    print(timeTaken)

    return


if __name__ == "__main__":
    main()
