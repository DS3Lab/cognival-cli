import collections
import numpy as np, scipy.stats as st

def extract_results_gaze(combinations):
    combination_results = collections.defaultdict(list)                 
    fold_errors = collections.defaultdict(list)                         
    embeddings = set()
    for x, y in combinations.items():                                   
        embeddings.add(y['wordEmbedding'])
        combination_results[y['feature']].append((y['wordEmbedding'], y['AVERAGE_MSE']))
        fold_errors[y['wordEmbedding']].extend(y['ALL_MSE']) 

    embeddings = list(embeddings)

    results_lists = {}
    avg_results = {}
    ci_results = {}
    for emb in embeddings:
        for res in combination_results.values():
            for r in res:
                if r[0] == emb:
                    if not emb in avg_results:
                        avg_results[emb] = [r[1]]
                    else:
                        avg_results[emb].append(r[1])
        results_lists[emb] = avg_results[emb]
        avg_results[emb] = sum(avg_results[emb]) / len(avg_results[emb])
        if len(results_lists[emb]) > 1:
            ci_results[emb] = st.t.interval(0.95, len(results_lists[emb])-1, loc=np.mean(results_lists[emb]), scale=st.sem(results_lists[emb]))
        else:
            ci_results[emb] = results_list[0], results_list[0]

    return fold_errors, results_lists, avg_results, ci_results
