import csv
import os
import subprocess

from pathlib import Path
import pandas as pd

def bonferroni_correction(alpha, no_hypotheses):
    return float(alpha / no_hypotheses)


def test_significance(baseline, model, alpha, test, debug=False):
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
    
    model = str(model).split('/')[-1]
    name = model.replace('.txt', '').replace('embeddings_avg_errors_', '')

    if debug:
        if "not significant" in str(output):
            print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
        else:
            print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))

    return significant, pvalue, name


def save_errors(emb_type, emb_scores, emb_filename, base_scores, base_filename, output_dir):
    emb_scores['error'] = emb_scores['error'].abs()

    if base_scores is not None:
        base_scores['error'] = base_scores['error'].abs()

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
