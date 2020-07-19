import collections

def extract_results(combinations):
    combination_results = collections.defaultdict(list)
    fold_errors = collections.defaultdict(list)
    for x, y in combinations.items():
        combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])
        try:
            fold_errors[y['wordEmbedding']].extend(y['ALL_MSE'])
        except KeyError:
            pass

    # average over sources:
    results_lists = {}
    avg_results = {}
    for emb, res in combination_results.items():
        results_lists[emb] = res
        avg_results[emb] = sum(res) / len(res)

    return fold_errors, results_lists, avg_results
