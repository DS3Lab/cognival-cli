import subprocess
import sig_test_config


def bonferroni_correction(alpha, no_hypotheses):
    return float(alpha / no_hypotheses)


def test_significance(baseline, model, alpha):
    command = ["python", "testSignificanceNLP/testSignificance.py", baseline, model, str(alpha), sig_test_config.test]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('>', output)
    output_str = output.decode('utf-8')
    pvalue = float(str(output).split(": ")[-1].replace("\\n'", ""))
    model = model.split('/')[-1]
    name = model.split('.')[0]
    if "not significant" in str(output):
        print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
    else:
        print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))

    return output_str, pvalue, name


def save_scores(emb_scores, emb_filename, base_scores, base_filename, output_dir, modality):
    """Save scores to temporary file. Compare embedding scores to baseline
    scores since word order and number of words differ."""

    emb_file = open(output_dir  / 'tmp' / modality / emb_filename, 'w')
    base_file = open(output_dir / 'tmp' / modality / base_filename, 'w')
    for word, score in emb_scores.items():
        # todo: absolute values or not?
        if word in base_scores:
            print(abs(float(score)), file=emb_file)
            print(abs(float(base_scores[word])), file=base_file)
