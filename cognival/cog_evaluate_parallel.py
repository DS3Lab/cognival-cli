import collections
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging
import signal
import sys
from multiprocessing import Pool
from datetime import  datetime
import cog_evaluate

from contextlib import redirect_stdout

from handlers.file_handler import *
from utils import animated_loading

import GPUtil
from termcolor import cprint, colored

#TODO: clear all WARNINGS!

class StopException(Exception): pass

def run_parallel(config_dict,
                 emb_to_random_dict,
                 embeddings_list,
                 cog_sources_list,
                 cog_source_to_feature,
                 n_jobs=None,
                 gpu_ids=None):

    if gpu_ids:
        gpu_ids_list = gpu_ids
    else:
        gpu_ids_list = []

    startTime = datetime.now()

    ##############################################################################
    #   OPTION GENERATION
    ##############################################################################
    cprint("\nParametrizing runs ... ", end = '')

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

    cprint("Done.")
    ##############################################################################
    #   JOINED DATAFRAMES GENERATION
    ##############################################################################

    ##############################################################################
    #   Parallelized version
    ##############################################################################

    print()
    cprint("Prediction", attrs=['bold', 'reverse'], color='green')
    print()

    if n_jobs:
        proc = n_jobs
    else:
        proc = min(os.cpu_count()-1, config_dict["n_proc"])
    
    cprint('Number of processes: ', color='cyan', end='')
    print(proc)
    if gpu_ids_list:
        cprint("Number of GPUs allocated: ", color="cyan", end='')
        print(len(gpu_ids_list))
        cprint("Allocated GPU IDs: ", color="cyan", end='')
        print(', '.join(map(str, gpu_ids_list)))
    print()
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes=proc)
    signal.signal(signal.SIGINT, original_sigint_handler)

    async_results_proper = []
    async_results_random = []
    rand_embeddings = []

    try:
        for option in options:
            random_embeddings = option["random_embedding"]
            rand_embeddings.append(random_embeddings)
            result_proper = pool.apply_async(cog_evaluate.run_single, args=('proper',
                                                                            config_dict,
                                                                            option["wordEmbedding"],
                                                                            option["cognitiveData"],
                                                                            option["modality"],
                                                                            option["feature"],
                                                                            option["truncate_first_line"],
                                                                            ','.join(map(str, gpu_ids_list))))
            
            result_random = []
            if random_embeddings:
                for random_embedding in config_dict["randEmbSetToParts"][random_embeddings]:
                    result_random.append(pool.apply_async(cog_evaluate.run_single, args=('random',
                                                                                        config_dict,
                                                                                        random_embedding,
                                                                                        option["cognitiveData"],
                                                                                        option["modality"],
                                                                                        option["feature"],
                                                                                        option["truncate_first_line"],
                                                                                        ','.join(map(str, gpu_ids_list)))))
                
            async_results_proper.append(result_proper)
            async_results_random.append(result_random)

        ar_monitoring = async_results_proper + [p for rand_list in async_results_random for p in rand_list]

        # Main loop
        while True:
            try:
                all_ready = True
                completed = 0
                for ar in ar_monitoring:
                    if not ar.ready():
                        all_ready = False
                    else:
                        # Terminate pool if a process fails
                        if not ar.successful():
                            print(ar.get())
                            pool.terminate()
                            raise StopException
                        else:
                            completed += 1

                if all_ready:
                    break

                f = io.StringIO()
                line_count = 0
                try:
                    with redirect_stdout(f):
                        GPUtil.showUtilization()
                    s = f.getvalue()
                    line_count = len(s.split('\n'))
                    sys.stdout.write(colored(s, 'yellow'))
                    sys.stdout.write("\n")
                except ValueError:
                    pass
                animated_loading(completed, len(ar_monitoring))
                sys.stdout.write("\033[F"*(line_count))
                sys.stdout.flush()
            except StopException:
                break   

    except KeyboardInterrupt:
        pool.terminate()
        return
    else:
        pool.close()

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
