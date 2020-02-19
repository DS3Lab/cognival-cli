from datetime import datetime
import numpy as np

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


def run_single(mode, config, word_embedding, cognitive_data, modality, feature, truncate_first_line, gpu_id):
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

    # Tensorflow configuration
    import tensorflow as tf
    from tensorflow.compat.v1.keras.backend import set_session, clear_session

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
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25
        tf_config.gpu_options.visible_device_list = str(gpu_id)
    
    session = tf.compat.v1.Session(config=tf_config)
    set_session(session)
    
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

    # Clean-up tf session
    session.close()
    clear_session()

    return word_embedding, logging, word_error, history

    