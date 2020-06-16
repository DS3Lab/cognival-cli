import collections
import numpy as np, scipy.stats as st

def extract_results(combinations):
    combination_results = collections.defaultdict(list)
    fold_errors = collections.defaultdict(list)
    for x, y in combinations.items():
        combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])
        fold_errors[y['wordEmbedding']].extend(y['ALL_MSE'])

    # average over sources:
    results_lists = {}
    avg_results = {}
    ci_results = {}
    for emb, res in combination_results.items():
        results_lists[emb] = res
        avg_results[emb] = sum(res) / len(res)
        if len(res) > 1:
            ci_results[emb] = st.t.interval(0.95, len(res)-1, loc=np.mean(res), scale=st.sem(res))
        else:
            ci_results[emb] = res[0], res[0]

    return fold_errors, results_lists, avg_results, ci_results
