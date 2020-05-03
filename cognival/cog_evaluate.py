from datetime import datetime
import numpy as np

#own modules
from handlers.data_handler import data_handler
from handlers.model_handler import model_handler
from handlers.file_handler import *

from termcolor import cprint

def handler(mode,
            config,
            stratified_sampling,
            balance,
            word_embedding,
            cognitive_data,
            feature,
            truncate_first_line,
            network,
            gpu_id,
            legacy):
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

    words_test, X_train, y_train, X_test, y_test = data_handler(mode,
                                                                config,
                                                                stratified_sampling,
                                                                balance,
                                                                word_embedding,
                                                                cognitive_data,
                                                                feature,
                                                                truncate_first_line)

    word_error, grids_result, mserrors = model_handler(word_embedding,
                                                       cognitive_data,
                                                       feature,
                                                       config["cogDataConfig"][cognitive_data]["wordEmbSpecifics"][word_embedding],
                                                       config['type'],
                                                       words_test,
                                                       X_train,
                                                       y_train,
                                                       X_test,
                                                       y_test,
                                                       network,
                                                       gpu_id,
                                                       legacy)

    return word_error, grids_result, mserrors


def run_single(mode,
               config,
               emb_type,
               word_embedding,
               cognitive_data,
               cognitive_parent,
               multi_hypothesis,
               multi_file,
               stratified_sampling,
               balance,
               modality,
               feature,
               truncate_first_line,
               gpu_id,
               network,
               legacy):
    '''
    Takes a configuration dictionary and keys for a word embedding and cognitive
    data source, runs model, logs results and prepares output for plotting.

    :param mode: Type of embeddings, either 'proper' or 'random'
    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    :param truncate_first_line: If the first line of the embedding file should be truncated (when containing meta data)
    :param gpu_ids: IDs of available GPUs
    '''
    ##############################################################################
    #   Create logging information
    ##############################################################################
    
    logging = {"folds":[]}

    logging["type"] = emb_type
    logging["wordEmbedding"] = word_embedding
    logging["cognitiveData"] = cognitive_data
    logging["cognitiveParent"] = cognitive_parent
    logging["multi_hypothesis"] = multi_hypothesis
    logging["multi_file"] = multi_file
    logging["modality"] = modality
    logging["details"] = ", ".join(["{}={}".format(measure, value) \
            for measure, value in [("stratified_sampling", stratified_sampling), ("balance", balance)] if value])
    logging["feature"] = feature

    ##############################################################################
    #   Run model
    ##############################################################################

    startTime = datetime.now()

    word_error, grids_result, mserrors = handler(mode,
                                                 config,
                                                 stratified_sampling,
                                                 balance,
                                                 word_embedding,
                                                 cognitive_data,
                                                 feature,
                                                 truncate_first_line,
                                                 network,
						 gpu_id,
                                                 legacy)

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
