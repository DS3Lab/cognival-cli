import collections

def extract_results_gaze(combinations):
    combination_results = collections.defaultdict(list)                 
    fold_errors = collections.defaultdict(list)                         
    embeddings = set()
    for x, y in combinations.items():                                   
        embeddings.add(y['wordEmbedding'])
        combination_results[y['feature']].append((y['wordEmbedding'], y['AVERAGE_MSE']))
        try:
            fold_errors[y['wordEmbedding']].extend(y['ALL_MSE']) 
        except KeyError:
            pass

    embeddings = list(embeddings)

    results_lists = {}
    avg_results = {}
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

    return fold_errors, results_lists, avg_results
