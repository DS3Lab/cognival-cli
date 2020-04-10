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


def save_errors(emb_scores, emb_filename, base_scores, base_filename, output_dir):
    emb_scores['error'] = emb_scores['error'].abs()
    base_scores['error'] = base_scores['error'].abs()

    words = []
    emb_scores_col = []
    base_scores_col = []

    for word in emb_scores.index:
        try:
            emb_score = emb_scores.loc[word]['error']
            base_score = base_scores.loc[word]['error']
            emb_scores_col.append(emb_score)
            base_scores_col.append(base_score)
            words.append(word)
        except KeyError:
            continue
    
    df_emb = pd.DataFrame({'error': emb_scores_col})
    df_base = pd.DataFrame({'error': base_scores_col})

    for df, out_file in [(df_emb, Path(output_dir) / emb_filename),
                         (df_base, Path(output_dir) / base_filename)]:
        df.to_csv(out_file,
                  sep=" ",
                  quotechar='"',
                  quoting=csv.QUOTE_NONNUMERIC,
                  doublequote=True,
                  encoding="utf-8",
                  header=False,
                  index=False) 
