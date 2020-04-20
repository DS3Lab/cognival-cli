import filecmp
import sys
from pathlib import Path

import pytest
import pandas as pd
sys.path.insert(0, 'cognival')

from handlers.data_handler import *

@pytest.fixture
def config():
    return {
    "PATH": "tests/",
    "cogDataConfig": {
        "zuco-eeg": {
            "dataset": "cognitive-data/eeg/zuco/zuco_scaled.txt",
            "features": [
                "ALL_DIM"
            ],
            "type": "multivariate_output",
            "wordEmbSpecifics": {
                "glove-50": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            26
                        ],
                        [
                            30
                        ],
                        [
                            20
                        ],
                        [
                            5
                        ]
                    ],
                    "validation_split": 0.2
                }
            }
        }
    },
    "n_proc": 40,
    "folds": 5,
    "outputDir": "zuco-feature-wordEmbedding",
    "seed": 123,
    "run_id": 11,
    "type":"word",
    "wordEmbConfig": {
        "glove-50": {
            "chunk_number": 4,
            "chunked": 1,
            "chunked_file": "glove-50",
            "chunk_ending": ".txt",
            "path": "reference/chunked_embeddings/glove.6B.50d.txt",
        }
    }
}


def test_chunker(tmpdir):
    refdir = Path('tests/reference')
    outdir = tmpdir.mkdir('output')
    chunk('tests/input',
            outdir,
            'glove.6B.50d.txt',
            'glove-50')
    assert not filecmp.dircmp(refdir, outdir).diff_files


def test_update():
    df_word_embedding = pd.read_csv('tests/reference/chunked_embeddings/glove-50_0.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE, index_col=0, names=['word', *['x{}'.format(idx) for idx in range(50)]])
    df_cognitive_data = pd.read_csv('tests/cognitive-data/eeg/zuco/zuco_scaled.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE)
    we_chunk_rows = df_word_embedding.shape[0] // 4
    df_word_embeddings_1 = df_word_embedding.iloc[:we_chunk_rows + 1, :]
    df_word_embeddings_2 = df_word_embedding.iloc[we_chunk_rows:(we_chunk_rows + 1)*2, :]
    df_join = pd.merge(df_cognitive_data, df_word_embeddings_1, how='left', on=['word'])
    # length w/o NANs after first merge
    assert len(df_join[df_join['x1'].notnull()]) == 3386

    update(df_join,
           df_word_embeddings_2,
           on_column=['word'],
           columns_to_omit=df_cognitive_data.shape[1],
           whole_row=True)

    # length w/O NANs after second merge
    assert len(df_join[df_join['x1'].notnull()]) == 3769

def test_update_synthetic_whole_row_True():
    #TODO: This appears to operate exactly in reverse (whole_row)
    df = pd.DataFrame({'A':[1,2,3,4],'B':[np.NaN,500,200,np.NaN],'C':[np.NaN,7.0,np.NaN,np.NaN]})
    df_new = pd.DataFrame({'A':[1,2,3,4],'B':[4,500,8,0],'C':[0,0,99,0]})

    update(df,
           df_new,
           on_column='A',
           columns_to_omit=1,
           whole_row=True)
    
    assert df.to_string() == '   A      B    C\n0  1    4.0  0.0\n1  2  500.0  7.0\n2  3  200.0  NaN\n3  4    0.0  0.0'


def test_update_synthetic_whole_row_False():
    #TODO: This appears to operate exactly in reverse (whole_row)
    df = pd.DataFrame({'A':[1,2,3,4],'B':[np.NaN,500,200,np.NaN],'C':[np.NaN,7.0,np.NaN,np.NaN]})
    df_new = pd.DataFrame({'A':[1,2,3,4],'B':[4,500,8,0],'C':[0,0,99,0]})

    update(df,
           df_new,
           on_column='A',
           columns_to_omit=1,
           whole_row=False)
    assert df.to_string() == '   A      B     C\n0  1    4.0   0.0\n1  2  500.0   7.0\n2  3  200.0  99.0\n3  4    0.0   0.0'


def test_dfMultiJoin():
    df_word_embedding = pd.read_csv('tests/input/glove.6B.50d.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE, index_col=0, names=['word', *['x{}'.format(idx) for idx in range(50)]])
    df_cognitive_data = pd.read_csv('tests/cognitive-data/eeg/zuco/zuco_scaled.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE)
    df_join = df_multi_join(df_cognitive_data,
                            df_word_embedding,
                            'word',
                            4)
    assert len(df_join[df_join['x1'].notnull()]) == 4162


def test_multiJoin(config):
    df_cognitive_data = pd.read_csv('tests/cognitive-data/eeg/zuco/zuco_scaled.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE)
    df_join = multi_join('proper',
                         config,
                         'word',
                         df_cognitive_data,
                         'eeg_zuco',
                         'ALL_DIM',
                         'glove-50')
    assert len(df_join[df_join['x_1'].notnull()]) == 4162


def test_dataHandler(config):
    #TODO: Handle chunked embeddings correctly
    #TODO: Ascertain: 75:25 Train test split, then 80:20 CV on train set, and 3-fold CV on train-train set?
    words_test, X_train, y_train, X_test, y_test = data_handler('proper',
                                                                config,
                                                                False,
                                                                False,
                                                               'glove-50',
                                                               'zuco-eeg',
                                                               'ALL_DIM',
                                                                False)
    assert [len(chunk) for chunk in X_train] == [3327, 3327, 3327, 3327, 3328]
    assert [len(chunk) for chunk in y_train] == [3327, 3327, 3327, 3327, 3328]
    assert [len(chunk) for chunk in words_test] == [832, 832, 832, 832, 831]
    assert [len(chunk) for chunk in X_test] == [832, 832, 832, 832, 831]
    assert [len(chunk) for chunk in y_test] == [832, 832, 832, 832, 831]


def test_split_folds(config):
    df_cognitive_data = pd.read_csv('tests/cognitive-data/eeg/zuco/zuco_scaled.txt', sep=" ", encoding="utf-8", quoting=csv.QUOTE_NONE)
    df_join = multi_join('proper',
                         config,
                         'word',
                         df_cognitive_data,
                         'eeg_zuco',
                         'ALL_DIM',
                         'glove-50')
    df_join.dropna(inplace=True)
    words = df_join['word']
    words = np.array(words, dtype='str').reshape(-1,1)
    df_join.drop(['word'], axis=1, inplace=True)
    features = df_cognitive_data.columns[1:]
    y = df_join[features]
    y = np.array(y, dtype='float')
    X = df_join.drop(features, axis=1)
    X = np.array(X, dtype='float')

    words_test, X_train, y_train, X_test, y_test = split_folds(words,
                                                               X,
                                                               y,
                                                               config['folds'],
                                                               config['seed'],
                                                               False,
                                                               None)
    
    assert [len(chunk) for chunk in X_train] == [3327, 3327, 3327, 3327, 3328]
    assert [len(chunk) for chunk in y_train] == [3327, 3327, 3327, 3327, 3328]
    assert [len(chunk) for chunk in words_test] == [832, 832, 832, 832, 831]
    assert [len(chunk) for chunk in X_test] == [832, 832, 832, 832, 831]
    assert [len(chunk) for chunk in y_test] == [832, 832, 832, 832, 831]
