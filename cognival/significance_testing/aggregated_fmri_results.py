import numpy as np, scipy.stats as st

def extract_results(combinations):
    combination_results = {}
    for x, y in combinations.items():
        if y['wordEmbedding'] not in combination_results:
            combination_results[y['wordEmbedding']] = [y['AVERAGE_MSE']]
        else:
            combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])

    # average over subjects:
    results_lists = {}
    avg_results = {}
    ci_results = {}
    for emb, res in combination_results.items():
        results_lists[emb] = res
        avg_results[emb] = sum(res) / len(res)
        ci_results[emb] = st.t.interval(0.95, len(res)-1, loc=np.mean(res), scale=st.sem(res))

    return results_lists, avg_results, ci_results
