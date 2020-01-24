import json

def aggregate_signi_fmri(result_dir,
                         test,
                         embeddings):

    significance = {}

    with open(result_dir / 'fmri' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if emb in experiment:
                    print(experiment)
                    hypotheses += 1
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1

            print(hypotheses)
            print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_eeg(result_dir,
                        test,
                        embeddings):

    significance = {}

    with open(result_dir / 'eeg' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)
        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if emb in experiment:
                    hypotheses += 1
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_gaze(result_dir,
                         test,
                         embeddings):
    
    significance = {}

    with open(result_dir / 'eye-tracking' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if emb in experiment:
                    hypotheses += 1
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance
