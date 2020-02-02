import json
from significance_testing import aggregate_significance


def extract_results(combinations):

    combination_results = {}
    for x, y in combinations.items():
        if y['wordEmbedding'] not in combination_results:
            combination_results[y['wordEmbedding']] = y['AVERAGE_MSE']

    return combination_results
