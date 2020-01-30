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

def run_parallel(config_dict,
         emb_to_random_dict,
         embeddings_list,
         cog_sources_list,
         cog_source_to_feature,
         cpu_count=None):
    startTime = datetime.now()

    ##############################################################################
    #   OPTION GENERATION
    ##############################################################################
    print("\nGenerating options ...")

    options = []
    #GENERATE all possible case scenarios:
    cog_data_list = [cognitive_data for cognitive_data in config_dict["cogDataConfig"] if cognitive_data in cog_sources_list]
    for idx, cognitive_data in enumerate(cog_data_list):
        feature_list = [feature for feature in config_dict["cogDataConfig"][cognitive_data]["features"] if feature in cog_source_to_feature[cog_data_list[idx]]]
        for feature in feature_list:
            for word_embedding in embeddings_list:
                truncate_first_line = config_dict["wordEmbConfig"][word_embedding]["truncate_first_line"]
                option = {"cognitiveData": "empty", "feature": "empty", "wordEmbedding": "empty"}
                option["cognitiveData"] = cognitive_data
                option["modality"] = config_dict["cogDataConfig"][cognitive_data]["modality"]
                option["feature"] = feature
                option["wordEmbedding"] = word_embedding
                option["random_embedding"] = emb_to_random_dict.get(word_embedding, None)
                option["truncate_first_line"] = truncate_first_line
                options.append(option)


    print("\nSuccessful options generation")

    ##############################################################################
    #   JOINED DATAFRAMES GENERATION
    ##############################################################################

    ##############################################################################
    #   Parallelized version
    ##############################################################################

    print("\nModels creation, fitting, prediction ...\n")

    if cpu_count:
        proc = cpu_count
    else:
        proc = min(os.cpu_count()-1, config_dict["cpu_count"])
    print("Running on " + str(proc) + " processors\n")
    pool = Pool(processes=proc)
    async_results_proper = []
    async_results_random = []
    rand_embeddings = []

    for option in options:
        random_embeddings = option["random_embedding"]
        rand_embeddings.append(random_embeddings)
        result_proper = pool.apply_async(cog_evaluate.run_single, args=('proper',
                                                                        config_dict,
                                                                        option["wordEmbedding"],
                                                                        option["cognitiveData"],
                                                                        option["modality"],
                                                                        option["feature"],
                                                                        option["truncate_first_line"]),)
        
        result_random = []
        if random_embeddings:
            for random_embedding in config_dict["randEmbSetToParts"][random_embeddings]:
                result_random.append(pool.apply_async(cog_evaluate.run_single, args=('random',
                                                                                    config_dict,
                                                                                    random_embedding,
                                                                                    option["cognitiveData"],
                                                                                    option["modality"],
                                                                                    option["feature"],
                                                                                    option["truncate_first_line"],)))
            
        async_results_proper.append(result_proper)
        async_results_random.append(result_random)

    pool.close()

    ar_monitoring = async_results_proper + [p for rand_list in async_results_random for p in rand_list]

    while (False in [ar.ready() == True for ar in ar_monitoring]):
        completed = [ar.ready() == True for ar in ar_monitoring].count(True)
        animated_loading(completed, len(ar_monitoring))

    pool.join()

    async_results_proper = [p.get() for p in async_results_proper]
    async_results_random = [[p.get() for p in p_list] for p_list in async_results_random]
    
    print("\nExecution complete. Time taken:")

    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))

    
    results_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for idx, option in enumerate(options):
        cog_source = option["modality"]
        results_dict[cog_source]["proper"].append(async_results_proper[idx])
        results_dict[cog_source]["random"].append(async_results_random[idx])
        results_dict[cog_source]["rand_embeddings"].append(rand_embeddings[idx])
        results_dict[cog_source]["options"].append(option)

    return results_dict
