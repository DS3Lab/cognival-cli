import subprocess

from pathlib import Path

def bonferroni_correction(alpha, no_hypotheses):
    return float(alpha / no_hypotheses)


def test_significance(baseline, model, alpha, test, debug=False):
    command = ["python",
               "significance_testing/testSignificanceNLP/testSignificance.py",
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
    name = model.replace('.txt', '').replace('embeddings_scores_', '')

    if debug:
        if "not significant" in str(output):
            print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
        else:
            print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))

    return significant, pvalue, name


def save_scores(emb_scores, emb_filename, base_scores, base_filename, output_dir, modality):
    """Save scores to temporary file. Compare embedding scores to baseline
    scores since word order and number of words differ."""

    emb_file = open(Path(output_dir) / emb_filename, 'w')
    base_file = open(Path(output_dir) / base_filename, 'w')
    for word, score in emb_scores.items():
        # todo: absolute values or not?
        if word in base_scores:
            print(abs(float(score)), file=emb_file)
            print(abs(float(base_scores[word])), file=base_file)
