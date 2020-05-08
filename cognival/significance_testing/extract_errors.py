import numpy as np
from termcolor import cprint
import pandas as pd

import csv
from pathlib import Path
from .testing_helpers import save_errors

def extract_errors(run_id, modality, experiment, mapping_dict, input_dir, results_dir):
    embedding_path = Path(input_dir) / mapping_dict[experiment]['proper']
    try:
        random_path = Path(input_dir) / mapping_dict[experiment]['random']
    except KeyError:
        random_path = ''
    emb_type = mapping_dict[experiment]['type']
    
    embeddings_df = pd.read_csv(embedding_path / '{}.txt'.format(mapping_dict[experiment]['embedding']),
				sep=" ",
				encoding="utf-8",
				quotechar='"',
				quoting=csv.QUOTE_NONNUMERIC,
				doublequote=True)

    try:
        rand_emb_file = mapping_dict[mapping_dict[experiment]['random_name']]['embedding']
        random_df = pd.read_csv(random_path / '{}.txt'.format(rand_emb_file),
                                    sep=" ",
                                    encoding="utf-8",
                                    quotechar='"',
                                    quoting=csv.QUOTE_NONNUMERIC,
                                    doublequote=True)
        rand_embs_available = True
    except (KeyError, FileNotFoundError):
        rand_embs_available = False
        #cprint('No results found for random embedding associated with {}, skipping ...'.format(experiment), 'red')

    embeddings_scores = {}
    baseline_scores = {}
    
    if modality in ('fmri', 'eeg'):
        embeddings_df.insert(1, 'error', embeddings_df[embeddings_df.columns.difference([emb_type])].mean(axis='columns'))
        embeddings_df.drop(embeddings_df.columns.difference([emb_type, 'error']), axis='columns', inplace=True)
        if rand_embs_available:
            random_df.insert(1, 'error', random_df[random_df.columns.difference([emb_type, 'error'])].mean(axis='columns'))
            random_df.drop(random_df.columns.difference([emb_type, 'error']), axis='columns', inplace=True)

    elif modality == 'eye-tracking':
        embeddings_df.rename(columns={embeddings_df.columns[1]:'error'}, inplace=True)
        if rand_embs_available:
            random_df.rename(columns={random_df.columns[1]:'error'}, inplace=True)

    embeddings_df.set_index(emb_type, inplace=True)
    if embeddings_df.index.has_duplicates:
        # Discard all sentences from zuco2 that already appear in zuco 1 (avoid distortion)
        cprint("Warning: {} duplicate rows found, only kept first occurrence.".format(len(embeddings_df.iloc[embeddings_df.index.duplicated()])), color='yellow')
        embeddings_df = embeddings_df.iloc[~embeddings_df.index.duplicated()]
 
        if rand_embs_available:
            random_df = random_df.iloc[~random_df.index.duplicated()]
	
    if rand_embs_available:
        random_df.set_index(emb_type, inplace=True)
        assert not random_df.index.has_duplicates
    else:
        random_df = None

    save_errors(emb_type,
                embeddings_df,
                'embeddings_avg_errors_' + '{}.txt'.format(experiment),
                random_df,
                'baseline_avg_errors_' + '{}.txt'.format(experiment),
                results_dir)

    if random_df is not None:
        return True
