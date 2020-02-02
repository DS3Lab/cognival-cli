import argparse
from datetime import datetime

from prompt_toolkit.shortcuts import ProgressBar
from termcolor import cprint


#own modules
from handlers.data_handler import data_handler
from handlers.model_handler import model_handler
from handlers.file_handler import *

def handler(mode, config, word_embedding, cognitive_data, feature, truncate_first_line):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, applies a model (as per configuration) and returns
    per-word errors, resulting grids and mean squared error (MSE).
    Wraps data_handler and model_handler.

    :param mode: Type of embeddings, either 'proper' or 'random'
    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    :param truncate_first_line: If the first line of the embedding file should be truncated (when containing meta data)
    '''

    words_test, X_train, y_train, X_test, y_test = data_handler(mode, config,word_embedding,cognitive_data,feature,truncate_first_line)
    word_error, grids_result, mserrors = model_handler(config["cogDataConfig"][cognitive_data]["wordEmbSpecifics"][word_embedding],
                                         words_test, X_train, y_train, X_test, y_test)

    return word_error, grids_result, mserrors


def run(config,
        word_embedding,
        random_embeddings,
        cognitive_data,
        feature,
        truncate_first_line,
        parallelized=False):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, runs model, logs results and prepares output for plotting.

    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param random_embeddings: String specifying corresponding random embedding or None
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    :param truncate_first_line: If the first line of the embedding file should be truncated (when containing meta data)
    '''
    cprint('Evaluating proper embedding {} ...'.format(word_embedding), 'cyan')
    results = []
    results.append(run_single('proper', config, word_embedding, cognitive_data, feature, truncate_first_line))
    if random_embeddings:
        cprint('Evaluating associated random embedding {} ...'.format(random_embeddings), 'cyan')
        if parallelized:
            for random_embedding in config["randEmbSetToParts"][random_embeddings]:
                results.append(run_single('random', config, random_embedding, cognitive_data, feature, truncate_first_line))
        else:
            with ProgressBar() as pb:
                for random_embedding in pb(config["randEmbSetToParts"][random_embeddings]):
                    results.append(run_single('random', config, random_embedding, cognitive_data, feature, truncate_first_line))
    return results


def run_single(mode, config, word_embedding, cognitive_data, modality, feature, truncate_first_line):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, runs model, logs results and prepares output for plotting.

    :param mode: Type of embeddings, either 'proper' or 'random'
    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    :param truncate_first_line: If the first line of the embedding file should be truncated (when containing meta data)
    '''

    ##############################################################################
    #   Create logging information
    ##############################################################################
    
    logging = {"folds":[]}

    logging["wordEmbedding"] = word_embedding
    logging["cognitiveData"] = cognitive_data
    logging["modality"] = modality
    logging["feature"] = feature

    ##############################################################################
    #   Run model
    ##############################################################################

    startTime = datetime.now()

    word_error, grids_result, mserrors = handler(mode, config, word_embedding, cognitive_data, feature, truncate_first_line)

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
            logging['folds'][i]['MSE_PREDICTION_ALL_DIM'] = list(mserrors[i])
            logging['folds'][i]['MSE_PREDICTION'] = np.mean(mserrors[i])
        elif config['cogDataConfig'][cognitive_data]['type'] == "single_output":
            logging['folds'][i]['MSE_PREDICTION'] = mserrors[i]

        logging['folds'][i]['LOSS'] = grids_result[i].best_estimator_.model.history.history['loss']
        logging['folds'][i]['VALIDATION_LOSS'] = grids_result[i].best_estimator_.model.history.history['val_loss']

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

    return word_embedding, logging, word_error, history

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

    start_time = datetime.now()

    logging, word_error, history = run(config, word_embedding, cognitive_data, feature)

    ##############################################################################
    #   Saving results
    ##############################################################################

    write_results(config, logging, word_error, history)

    time_taken = datetime.now() - start_time
    print(time_taken)

    return
    