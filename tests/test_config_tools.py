import json
import pytest
import sys

sys.path.insert(0, '..')

from sideCode.configJsonGenerator import *
from sideCode.configJsonEditor import *

@pytest.fixture
def conf_json_gen_reference():
    return {
                "cognitiveData": {
                    "mitchell-1000-0": {
                        "features": [
                            "ALL_DIM"
                        ]
                    },
                    "mitchell-1000-1": {
                        "features": [
                            "ALL_DIM"
                        ]
                    },
                    "mitchell-1000-2": {
                        "features": [
                            "ALL_DIM"
                        ]
                    }
                },
                "configFile": "config/setupConfig.json",
                "wordEmbeddings": [
                    "word2vec",
                    "fasttext-crawl",
                    "fasttext-crawl-subword",
                    "fasttext-wiki-news",
                    "fasttext-wiki-news-subword",
                    "elmo",
                    "bert-base",
                    "bert-large",
                    "bert-service-base",
                    "bert-service-large",
                    "glove-50",
                    "glove-100",
                    "glove-200",
                    "glove-300",
                    "random-embeddings-100",
                    "random-embeddings-1024",
                    "random-embeddings-200",
                    "random-embeddings-300",
                    "random-embeddings-50",
                    "random-embeddings-768",
                    "random-embeddings-850",
                    "wordnet2vec"
                ]
            }


@pytest.fixture
def conf_json_editor_reference():
    return {
            "PATH": "/home/adrian/repos/cognival/",
            "cogDataConfig": {
                "zuco-eeg": {
                    "dataset": "cognitive-data/eeg/zuco/zuco_scaled.txt",
                    "features": [
                        "ALL_DIM"
                    ],
                    "type": "multivariate_output",
                    "wordEmbSpecifics": {
                        "bert-service-base": {
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
                                    400
                                ],
                                [
                                    200
                                ]
                            ],
                            "validation_split": 0.2
                        },
                        "bert-service-large": {
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
                                    600
                                ],
                                [
                                    200
                                ]
                            ],
                            "validation_split": 0.2
                        },
                        "glove-100": {
                            "activations": [
                                "relu"
                            ],
                            "batch_size": [
                                128
                            ],
                            "cv_split": 3,
                            "epochs": [
                                30
                            ],
                            "layers": [
                                [
                                    50
                                ],
                                [
                                    30
                                ]
                            ],
                            "validation_split": 0.2
                        },
                        "glove-200": {
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
                                    100
                                ],
                                [
                                    50
                                ]
                            ],
                            "validation_split": 0.2
                        },
                        "glove-300": {
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
                                    50
                                ],
                                [
                                    150
                                ]
                            ],
                            "validation_split": 0.2
                        },
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
            "cpu_count": 2,
            "folds": 5,
            "outputDir": "zuco-feature-wordEmbedding",
            "seed": 123,
            "version": 23.0,
            "wordEmbConfig": {
                "glove-100": {
                    "chunked": 0,
                    "path": "embeddings/glove-6B/glove.6B.100d.txt"
                },
                "glove-200": {
                    "chunked": 0,
                    "path": "embeddings/glove-6B/glove.6B.200d.txt"
                },
                "glove-300": {
                    "chunked": 0,
                    "path": "embeddings/glove-6B/glove.6B.300d.txt"
                },
                "glove-50": {
                    "chunked": 0,
                    "path": "embeddings/glove-6B/glove.6B.50d.txt"
                }
            }
        }


def test_generate_config_json(conf_json_gen_reference, tmpdir):
    tmpdir.mkdir('output')
    generate_config_json(path=tmpdir / 'output',
                         cd='mitchell',
                         dim=1000,
                         start=0,
                         end=2)
    with open(tmpdir / 'output' / 'mitchell-1000.json') as f:
        conf_json_gen_output = json.load(f)
    conf_json_gen_output == conf_json_gen_reference


def test_editor_config_json(conf_json_editor_reference, tmpdir):
    embedding_parameters = {'bert-service-base': {"activations": [
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
                                        400
                                    ],
                                    [
                                        200
                                    ]
                                ],
                                "validation_split": 0.2
                            },
                            'bert-service-large': {
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
                                    600
                                ],
                                [
                                    200
                                ]
                            ],
                            "validation_split": 0.2
                            }
                        }
    tmpdir.mkdir('output')
    editor_config_json(embedding_parameters=embedding_parameters,
                       input_path=Path("../config/"),
                       output_path=tmpdir / 'output',
                       configFile='example_1.json')
    with open(tmpdir / 'output' / 'example_1.json') as f:
        conf_json_editor_output = json.load(f)
        assert conf_json_editor_output == conf_json_editor_reference