import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging
import sys
from multiprocessing import Pool
from datetime import  datetime
import cog_evaluate
from handlers.file_handler import get_config
from handlers.file_handler import *
from utils import animated_loading
#TODO: clear all WARNINGS!

def main(controller_config):

    startTime = datetime.now()

    with open(controller_config, 'r') as f:
        data = json.load(f)

    config = get_config(data["configFile"])

    ##############################################################################
    #   OPTION GENERATION
    ##############################################################################

    print("\nGENERATING OPTIONS...")

    options = []
    #GENERATE all possible case scenarios:
    for cognitive_data in data["cognitiveData"]:
        for feature in data["cognitiveData"][cognitive_data]["features"]:
            for word_embedding in data["wordEmbeddings"]:
                option = {"cognitiveData": "empty", "feature": "empty", "wordEmbedding": "empty"}
                option["cognitiveData"] = cognitive_data
                option["feature"] = feature
                option["wordEmbedding"] = word_embedding
                options.append(option)

    loggings = []
    word_errors = []
    histories = []

    print("\nSUCCESSFUL OPTIONS GENERATION")

    ##############################################################################
    #   JOINED DATAFRAMES GENERATION
    ##############################################################################

    ##############################################################################
    #   Parallelized version
    ##############################################################################

    print("\nMODELS CREATION, FITTING, PREDICTION...\n ")

    proc = min(os.cpu_count()-1, config["cpu_count"])
    print("RUNNING ON " + str(proc) + " PROCESSORS\n")
    pool = Pool(processes=proc)
    async_results = [pool.apply_async(cog_evaluate.run,args=(config,
                                           options[i]["wordEmbedding"],
                                           options[i]["cognitiveData"],
                                           options[i]["feature"])) for i in range(len(options))]
    pool.close()

    while (False in [async_results[i].ready() == True for i in range(len(async_results))]):
        completed = [async_results[i].ready() == True for i in range(len(async_results))].count(True)
        animated_loading(completed, len(async_results))

    pool.join()

    for p in async_results:
        logging, word_error, history = p.get()
        loggings.append(logging)
        word_errors.append(word_error)
        histories.append(history)

    print("\nSUCCESSFUL MODELS")

    ##############################################################################
    #   Store results
    ##############################################################################

    print("\nSTORING RESULTS...")

    all_runs = {}

    for i in range(0,len(loggings)):
        config = get_config(data["configFile"])
        write_results(config, loggings[i], word_errors[i], histories[i])
        options[i]["AVERAGE_MSE"] = loggings[i]["AVERAGE_MSE"]
        all_runs[config["version"]] = options[i]
        update_version(data["configFile"])

    write_options(config, all_runs)

    print("\nSUCCESSFUL STORING")

    print("\nSUCCESSFUL RUN")

    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))

    return


if __name__=="__main__":
    main("config/c_single_random.json")