import json
import sig_test_config


def aggregate_signi_fmri(result_dir,
                         test,
                         embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl_',
                                       'fasttext-wiki-news_',
                                       'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                                       'bert-service-large', 'elmo'],
                        only_1000_voxels=True):

    significance = {}

    with open(result_dir / 'fmri' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                # take only results from 1000 voxels
                if emb in p and not only_1000_voxels or ('-1000-' in p or 'brennan' in p):
                    print(p)
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            print(hypotheses)
            print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_eeg(result_dir,
                        test,
                        embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl_',
                                      'fasttext-wiki-news_',
                                      'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                                      'bert-service-large', 'elmo']):

    significance = {}

    with open(result_dir / 'eeg' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                if emb in p:
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_gaze(result_dir,
                         test,
                         embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec',
                                       'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                                       'bert-service-large', 'elmo']):
    
    significance = {}

    with open(result_dir / 'gaze' / '{}.json'.format(test)) as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                if emb in p:
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance
