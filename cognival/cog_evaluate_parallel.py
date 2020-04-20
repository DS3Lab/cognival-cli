import collections
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging
import signal
import sys
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
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
                option["type"] = config_dict["type"]
                option["cognitiveData"] = cognitive_data
                option["cognitiveParent"] = config_dict["cogDataConfig"][cognitive_data]["parent"]
                option["multiHypothesis"] = config_dict["cogDataConfig"][cognitive_data]["multi_hypothesis"]  
                option["multiFile"] = config_dict["cogDataConfig"][cognitive_data]["multi_file"]
                option["stratifiedSampling"]  = config_dict["cogDataConfig"][cognitive_data]["stratified_sampling"]
                option["balance"]  = config_dict["cogDataConfig"][cognitive_data]["balance"]
                option["modality"] = config_dict["cogDataConfig"][cognitive_data]["modality"]
                option["feature"] = feature
                option["wordEmbedding"] = word_embedding if word_embedding in config_dict["cogDataConfig"][cognitive_data]['wordEmbSpecifics'] else None
                if not option['wordEmbedding']:
                    continue
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
            proper_params = [('proper',
                                                                            config_dict,
                                                                            option["type"],
                                                                            option["wordEmbedding"],
                                                                            option["cognitiveData"],
                                                                            option["cognitiveParent"],
                                                                            option["multiHypothesis"],
                                                                            option["multiFile"],
                                                                            option["stratifiedSampling"],
                                                                            option["balance"],
                                                                            option["modality"],
                                                                            option["feature"],
                                                                            option["truncate_first_line"],
                                                                            ','.join(map(str, gpu_ids_list)))]
            random_params = [] 
            if random_embeddings:
                for random_embedding in config_dict["randEmbSetToParts"][random_embeddings]:
                    random_params.append(('random',
                                                                                        config_dict,
                                                                                        option["type"],
                                                                                        random_embedding,
                                                                                        option["cognitiveData"],
                                                                                        option["cognitiveParent"],
                                                                                        option["multiHypothesis"],
                                                                                        option["multiFile"],
                                                                                        option["stratifiedSampling"],
                                                                                        option["balance"],
                                                                                        option["modality"],
                                                                                        option["feature"],
                                                                                        option["truncate_first_line"],
                                                                                        ','.join(map(str, gpu_ids_list))))
                
            result_proper = pool.starmap_async(cog_evaluate.run_single, proper_params)
            if random_params:
                result_random = pool.starmap_async(cog_evaluate.run_single, random_params)
            else:
                result_random = None
            async_results_proper.append(result_proper)
            async_results_random.append(result_random)
        
        async_results_proper = [(idx, ar) for idx, ar in enumerate(async_results_proper)]
        async_results_random = [(idx, ar) for idx, ar in enumerate(async_results_random)]

        prev_completed = None
    
        # Main loop

        num_jobs = len(async_results_proper)
        completed = 0

        collector = collections.defaultdict(lambda: collections.defaultdict(dict))
        next_yield = 0
        while True:
            try:
                async_results_proper_remaining = []
                async_results_random_remaining = []
                all_ready = True

                # Iterate over MapResult objects for proper and random embeddings
                for type_, ar_list, ar_list_remaining in [('proper', async_results_proper, async_results_proper_remaining),
                                                          ('random', async_results_random, async_results_random_remaining)]:
                    for idx, ar in ar_list:
                        if ar is not None:
                            # Process not yet terminated
                            if not ar.ready():
                                all_ready = False
                                ar_list_remaining.append((idx, ar))
                            # Processed finished or failed
                            else:
                                # Terminate pool if a process fails
                                if not ar.successful():
                                    print(ar.get())
                                    pool.terminate()
                                    raise StopException
                                else:
                                    if type_ == 'proper':
                                        collector[idx]['proper'] = ar.get()[0]
                                    elif type_ == 'random':
                                        collector[idx]['random'] = ar.get()

                # Iterate over collector
                for idx in sorted(collector.keys()):
                    # As soon as results for both proper and random embeddings available: Yield results and delete
                    if idx == next_yield and (collector[idx]['proper'] and collector[idx]['random']):
                        
                        yield options[idx]['modality'], \
                              collector[idx]['proper'], \
                              collector[idx]['random'], \
                              rand_embeddings[idx], \
                              options[idx]

                        del collector[idx]
                        completed += 1                           
                        next_yield += 1

                if all_ready:
                    raise StopException

                f = io.StringIO()
                line_count = 0

                try:
                    # Show GPU utilization if any GPUs assigned
                    if gpu_ids_list:
                        with redirect_stdout(f):
                            GPUtil.showUtilization()
                        s = f.getvalue()
                        line_count = len(s.split('\n'))
                        sys.stdout.write(colored(s, 'yellow'))
                        sys.stdout.write("\n")
                except ValueError:
                    pass

                if completed != prev_completed:
                    cprint('Progress: #{}/{} ({:.2f}%)'.format(completed,
                                                              num_jobs,
                                                              (completed/num_jobs)*100),
                                                              attrs=['bold'])
                    prev_completed = completed

                async_results_proper = async_results_proper_remaining
                async_results_random = async_results_random_remaining

            except StopException:
                break   

    except KeyboardInterrupt:
        pool.terminate()
        raise
    else:
        pool.close()

    pool.join()

    print("\nExecution complete. Time taken:")

    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))
