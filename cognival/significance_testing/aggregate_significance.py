import json

def aggregate_signi_fmri(result_dir,
                         run_id,
                         test,
                         embeddings):

    significance = {}

    with open(result_dir / 'fmri' / str(run_id) / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if experiment.endswith(emb):
                    hypotheses += 1
                    corrected_alpha = data['hypotheses'][experiment]['bonferroni_alpha']
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_eeg(result_dir,
                        run_id,
                        test,
                        embeddings):

    significance = {}

    with open(result_dir / 'eeg' / str(run_id) / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if experiment.endswith(emb):
                    hypotheses += 1
                    corrected_alpha = data['hypotheses'][experiment]['bonferroni_alpha']
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_gaze(result_dir,
                         run_id,
                         test,
                         embeddings):
    
    significance = {}

    with open(result_dir / 'eye-tracking' / str(run_id) / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for experiment in data['hypotheses']:
                if experiment.endswith(emb):
                    hypotheses += 1
                    corrected_alpha = data['hypotheses'][experiment]['bonferroni_alpha']
                    if data['hypotheses'][experiment]['p_value'] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance
