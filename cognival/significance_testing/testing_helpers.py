import csv
import os
import subprocess

from pathlib import Path
import pandas as pd

import numpy as np
from scipy import stats
from permute.core import one_sample

def bonferroni_correction(alpha, no_hypotheses):
    return float(alpha / no_hypotheses)


def test_significance(baseline, model, alpha, test, debug=False):
    name = str(model).split('/')[-1].replace('.txt', '').replace('embeddings_avg_errors_', '')
    
    try:
        model = pd.read_csv(model,
                                 sep=" ",
                                 encoding="utf-8",
                                 quotechar='"',
                                 quoting=csv.QUOTE_NONNUMERIC,
                                 doublequote=True)

        baseline = pd.read_csv(baseline,
                                 sep=" ",
                                 encoding="utf-8",
                                 quotechar='"',
                                 quoting=csv.QUOTE_NONNUMERIC,
                                 doublequote=True)

        # Invert errors (smaller is better -> larger is better)
        model = 1.0 - model['error'].to_numpy()
        baseline = 1.0 - baseline['error'].to_numpy()

        if test == 'Permutation':
            pvalue, _ = one_sample(model, baseline, stat='mean')
        elif test == 'Wilcoxon':
            wilcoxon_results = stats.wilcoxon(model, baseline)
            pvalue = wilcoxon_results[1]
        else:
            raise NotImplementedError(test)

        if (float(pvalue) <= float(alpha)):
            significant = True
        else:
            significant = False

    except ValueError:
        raise ValueError("testSignificance has returned: {}".format(repr(output_str)))
    
    if debug:
        if not significant:
            print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
        else:
            print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))

    return significant, pvalue, name


def save_errors(emb_type, emb_scores, emb_filename, base_scores, base_filename, output_dir):
    strings = []
    emb_scores_col = []
    base_scores_col = []

    for string in emb_scores.index:
        try:
            emb_score = emb_scores.loc[string]['error']
            emb_scores_col.append(emb_score)
            if base_scores is not None:
                base_score = base_scores.loc[string]['error']
                base_scores_col.append(base_score)
            strings.append(string)
        except KeyError:
            continue
    
    df_emb = pd.DataFrame({emb_type: strings, 'error': emb_scores_col})
    
    if base_scores is not None:
        df_base = pd.DataFrame({emb_type: strings, 'error': base_scores_col})
        df_list = [(df_emb, Path(output_dir) / emb_filename),
                   (df_base, Path(output_dir) / base_filename)]
    else:
        df_list = [(df_emb, Path(output_dir) / emb_filename)]

    for df, out_file in df_list:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        df.to_csv(out_file,
                  sep=" ",
                  quotechar='"',
                  quoting=csv.QUOTE_NONNUMERIC,
                  doublequote=True,
                  encoding="utf-8",
                  header=True,
                  index=False) 
