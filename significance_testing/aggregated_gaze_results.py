import json
import numpy as np
import matplotlib.pyplot as plt
from significance_testing import aggregate_significance

def extract_results_gaze(combinations,
                         baselines,
                         embeddings):
    combination_results = {}
    for y in combinations.values():
        if y['feature'] not in combination_results:
            combination_results[y['feature']] = [(y['wordEmbedding'], y['AVERAGE_MSE'])]
        else:
            combination_results[y['feature']].append((y['wordEmbedding'], y['AVERAGE_MSE']))

    avg_results = {}
    for emb_type in embeddings + baselines:
        for res in combination_results.values():
            for r in res:
                if r[0] == emb_type:
                    if not emb_type in avg_results:
                        avg_results[emb_type] = [r[1]]
                    else:
                        avg_results[emb_type].append(r[1])
        avg_results[emb_type] = sum(avg_results[emb_type]) / len(avg_results[emb_type])

    return avg_results
