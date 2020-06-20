import csv
import os
import subprocess

from pathlib import Path
import pandas as pd

from mcpt import permutation_test as mc_permutation_test
import numpy as np

def iqr(x):
   q3 = np.quantile(x, .75)
   q1 = np.quantile(x, .25)
   return q3 - q1

def bonferroni_correction(alpha, no_hypotheses):
    return float(alpha / no_hypotheses)


def test_significance(baseline, model, alpha, test, debug=False):
    name = str(model).split('/')[-1].replace('.txt', '').replace('embeddings_avg_errors_', '')
    if not test.startswith('Permutation'):
        command = ["python",
                   str(Path(os.path.dirname(__file__)) / "testSignificanceNLP/testSignificance.py"),
                   baseline,
                   model,
                   str(alpha),
                   test]

        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
        output_str = output.decode('utf-8')
        
        try:
            result, pvalue_raw = output_str.split(": ")
            if 'is not significant' in result:
                significant = False
            elif 'is significant' in result:
                significant = True
            else:
                raise ValueError()
            pvalue = float(pvalue_raw.replace("\\n'", ""))
        except ValueError:
            raise ValueError("testSignificance has returned: {}".format(repr(output_str)))
        
        if debug:
            if "not significant" in str(output):
                print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
            else:
                print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))
    else:
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

        model = model['error'].to_numpy()
        baseline = baseline['error'].to_numpy()
        #print('Sum of absolute errors averaged across dimensions - model: {:.2f} | baseline: {:.2f}'.format(sum(model), sum(baseline)))
        # Take upper bound for p-value (worst-case), fixed confidence of 0.99
        if test == 'Permutation-Mean':
            pvalue = mc_permutation_test(model, baseline, n=25000, f='mean', side='both', cores=24, confidence=.99).upper
        elif test == 'Permutation-Median':
            pvalue = mc_permutation_test(model, baseline, n=25000, f='median', side='both', cores=24, confidence=.99).upper
        elif test == 'Permutation-IQR':
            pvalue = mc_permutation_test(model, baseline, n=25000, f=iqr, side='both', cores=24, confidence=.99).upper
        else:
            raise ValueError("test")
        significant = True if float(pvalue) <= float(alpha) else False

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
