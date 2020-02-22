import json
import os
import numpy as np

from pathlib import Path
from handlers.plot_handler import plot_handler

def update_run_id(configFile):
    '''
    Increments configuration run_id by 1
    '''
    with open(configFile, 'r') as fileReader:
        config = json.load(fileReader)

    config['run_id'] = config['run_id'] + 1

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

    output_dir = Path(config['PATH']) / config['outputDir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mapping_path = output_dir / 'mapping_{}.json'.format(config["run_id"])
    title = log["cognitiveData"] + '_' + log["feature"] + '_' + log["wordEmbedding"] + '_' + str(config["run_id"])
    path = Path(log["modality"]) / log["cognitiveData"] / log["feature"] / log["wordEmbedding"] / str(config["run_id"])
    rand_emb = None
    if not log["wordEmbedding"].startswith('random'):
        rand_emb = config['wordEmbConfig'][log['wordEmbedding']]['random_embedding']
    
        if rand_emb:
            rand_path = Path(log["modality"]) / log["cognitiveData"] / log["feature"] / rand_emb / str(config["run_id"])

    # Mapping dict patch
    mapping_key = "{}_{}_{}".format(log["cognitiveData"], log["feature"], log["wordEmbedding"])
    map_dict_patch = {mapping_key: {'embedding': log["wordEmbedding"],
                                     'cognitive-source': log["cognitiveData"],
                                     'cognitive-parent': log["cognitiveParent"],
                                     'modality': log["modality"],
                                     'feature': log["feature"],
                                     'proper': str(path),
                                     }}
    if rand_emb:
        map_dict_patch[mapping_key]['random'] = str(rand_path)
        map_dict_patch[mapping_key]['random_name'] = "{}_{}_{}".format(log["cognitiveData"], log["feature"], rand_emb)

    if not os.path.exists(mapping_path):
        with open(mapping_path, 'w') as f:
            json.dump(map_dict_patch, f, indent=4)
    else:
        with open(mapping_path,'r') as f:
            mapping_dict = json.load(f)
            mapping_dict.update(map_dict_patch)
        with open(mapping_path, 'w') as f:
            json.dump(dict(mapping_dict), f, indent=4)

    experiments_dir = output_dir / "experiments" / path

    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    with open(experiments_dir / '{}.json'.format(log["wordEmbedding"]),'w') as f:
        json.dump(log,f,indent=4, sort_keys=True)

    np.savetxt(experiments_dir / '{}.txt'.format(log["wordEmbedding"]), word_error, delimiter=" ", fmt="%s")
    
    if history:
        plot_handler(title, history, log, str(experiments_dir))

    return


def write_options(config, modality, run_stats):
    '''
    Writes summary information JSONs for given
    experimental runs, for subsequent significance testing

    :param configuration: dictionary
    :param run_stats: Stats and params for each experimental run
    '''

    outputDir = Path(config['PATH']) / config['outputDir']

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    with open(outputDir / "experiments" / modality / "options_{}.json".format(str(config["run_id"])), 'w') as fileWriter:
        json.dump(run_stats, fileWriter, indent=4, sort_keys=True)

    return
