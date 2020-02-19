def extract_results(combinations):
    combination_results = {}
    for x, y in combinations.items():
        if y['wordEmbedding'] not in combination_results:
            combination_results[y['wordEmbedding']] = [y['AVERAGE_MSE']]
        else:
            combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])

    # average over subjects:
    avg_results = {}
    for emb, res in combination_results.items():
        avg_results[emb] = sum(res) / len(res)

    return avg_results
