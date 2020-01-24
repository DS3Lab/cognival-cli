import json
from significance_testing import aggregate_significance


def extract_results(combinations,
                    baselines,
                    embeddings):
    combination_results = {}
    for x, y in combinations.items():
        # print(y['feature'], y['wordEmbedding'])
        if y['wordEmbedding'] not in combination_results:
            combination_results[y['wordEmbedding']] = [y['AVERAGE_MSE']]
        else:
            combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])

    # average over subjects:
    avg_results = {}
    for emb, res in combination_results.items():
        avg_results[emb] = sum(res) / len(res)

    return avg_results


def main():
    cognitive_dataset = 'wehbe'
    print("Evaluating combination:", )
    results = extract_results(cognitive_dataset)

    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl-subword',
                  'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec', 'bert-service-large', 'elmo']
    baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300',
                 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-768',
                 'random-embeddings-850', 'random-embeddings-1024', 'random-embeddings-1024']

    significance = aggregate_significance.aggregate_signi_fmri()

    for idx, (emb, base) in enumerate(zip(embeddings, baselines)):
        avg_base = results[base]
        avg_emb = results[emb]
        print(emb, avg_base, avg_emb, significance[emb])


if __name__ == '__main__':
    main()
