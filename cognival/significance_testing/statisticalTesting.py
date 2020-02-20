import numpy as np
from termcolor import cprint

from pathlib import Path
from .testing_helpers import save_scores

def extract_results(run_id, modality, experiment, mapping_dict, input_dir, results_dir):
    embedding_path = Path(input_dir) / mapping_dict[experiment]['proper']
    random_path = Path(input_dir) / mapping_dict[experiment]['random']
    
    embeddings_file = open(embedding_path / '{}.txt'.format(mapping_dict[experiment]['embedding']), 'r').readlines()
    try:
        rand_emb_file = mapping_dict[mapping_dict[experiment]['random_name']]['embedding']
        baseline_file = open(random_path / '{}.txt'.format(rand_emb_file), 'r').readlines()
    except FileNotFoundError:
        cprint('No results found for random embedding associated with {}, skipping ...'.format(experiment), 'red')
        return

    embeddings_scores = {}
    baseline_scores = {}
    
    if modality == 'fmri':
        for line in embeddings_file[1:]:
            line = line.strip().split()
            voxels = [float(i) for i in line[1:]]
            avg_voxels = np.mean(voxels)
            embeddings_scores[line[0]] = avg_voxels

        for line in baseline_file[1:]:
            line = line.strip().split()
            voxels = [float(i) for i in line[1:]]
            avg_voxels = np.mean(voxels)
            baseline_scores[line[0]] = avg_voxels

    if modality == 'eeg':
        for line in embeddings_file[1:]:
            line = line.strip().split()
            electrodes = [float(i) for i in line[1:]]
            avg_electrodes = np.mean(electrodes)
            embeddings_scores[line[0]] = avg_electrodes
    
        for line in baseline_file[1:]:
            line = line.strip().split()
            electrodes = [float(i) for i in line[1:]]
            avg_electrodes = np.mean(electrodes)
            baseline_scores[line[0]] = avg_electrodes

    if modality == 'eye-tracking':
        embeddings_scores = dict(line.strip().split() for line in embeddings_file[1:])
        baseline_scores = dict(line.strip().split() for line in baseline_file[1:])

    save_scores(embeddings_scores,
                'embeddings_scores_' + '{}.txt'.format(experiment),
                baseline_scores,
                'baseline_scores_' + '{}.txt'.format(experiment),
                results_dir,
                modality)