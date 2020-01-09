import json
import os
import numpy as np
from handlers.plot_handler import plot_handler

def update_version(configFile):
    '''
    Increments configuration version by 1
    '''
    with open(configFile, 'r') as fileReader:
        config = json.load(fileReader)

    config['version'] = config['version'] + 1

    with open(configFile,'w') as fileWriter:
        json.dump(config,fileWriter, indent=4, sort_keys=True)

    return config


def get_config(configFile):
    '''
    Loads configuration JSON as dictionary
    '''
    with open(configFile, 'r') as fileReader:
        config = json.load(fileReader)

    return config


def write_results(config, log, word_error, history):
    '''
    Writes experimental results (log) and word errors.

    :param config: configuration dictionary
    :param log: log dictionary
    :param word_error: np.array with word errors
    :param history: np.array with training history (loss)
    '''
    if not os.path.exists(config['outputDir']):
        os.mkdir(config['outputDir'])
    
    title = log["cognitiveData"] + '_' + log["feature"] + '_' + log["wordEmbedding"]+'_'+str(config["version"])

    output_dir = config['outputDir'] + "/" + title
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_dir+"/"+title+'.json','w') as f:
        json.dump(log,f,indent=4, sort_keys=True)

    np.savetxt(output_dir + "/" + title + '.txt', word_error, delimiter=" ", fmt="%s")
    
    if history:
        plot_handler(title, history, output_dir)
    else:
        print("No history, no plot ...")

    return


def write_options(config, run_stats):
    '''
    Writes summary information JSONs for given
    experimental runs, for subsequent significance testing

    :param configuration: dictionary
    :param run_stats: Stats and params for each experimental run
    '''

    outputDir = config['outputDir']

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    with open(outputDir+"/options"+str(config["version"])+'.json','w') as fileWriter:
        json.dump(run_stats, fileWriter, indent=4,sort_keys=True)

    return
