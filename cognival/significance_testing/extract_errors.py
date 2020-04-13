import numpy as np
from termcolor import cprint
import pandas as pd

import csv
from pathlib import Path
from .testing_helpers import save_errors

def extract_errors(run_id, modality, experiment, mapping_dict, input_dir, results_dir):
    embedding_path = Path(input_dir) / mapping_dict[experiment]['proper']
    random_path = Path(input_dir) / mapping_dict[experiment]['random']
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
       
    except FileNotFoundError:
        cprint('No results found for random embedding associated with {}, skipping ...'.format(experiment), 'red')
        return

    embeddings_scores = {}
    baseline_scores = {}
    
    if modality in ('fmri', 'eeg'):
        embeddings_df.insert(1, 'error', embeddings_df[embeddings_df.columns.difference([emb_type])].mean(axis='columns'))
        embeddings_df.drop(embeddings_df.columns.difference([emb_type, 'error']), axis='columns', inplace=True)
        random_df.insert(1, 'error', random_df[random_df.columns.difference([emb_type, 'error'])].mean(axis='columns'))
        random_df.drop(random_df.columns.difference([emb_type, 'error']), axis='columns', inplace=True)

    elif modality == 'eye-tracking':
        embeddings_df.rename(columns={embeddings_df.columns[0]:'error'}, inplace=True)
        random_df.rename(columns={random_df.columns[0]:'error'}, inplace=True)

    embeddings_df.set_index(emb_type, inplace=True)
    random_df.set_index(emb_type, inplace=True)

    assert not embeddings_df.index.has_duplicates
    assert not random_df.index.has_duplicates

    save_errors(embeddings_df,
                'embeddings_avg_errors_' + '{}.txt'.format(experiment),
                random_df,
                'baseline_avg_errors_' + '{}.txt'.format(experiment),
                results_dir)
