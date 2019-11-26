import json
from pathlib import Path

def editor_config_json(embedding_parameters=None,
                       input_path=Path("../config/"),
                       output_path=Path("."),
                       configFile='example_1.json'):
    with open(input_path / configFile,'r') as fR:
        setup = json.load(fR)

    for elem in setup["cogDataConfig"]:
        setup["cogDataConfig"][elem]["wordEmbSpecifics"]["glove-50"] = {
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

        
        for name, parameters in embedding_parameters.items():
            setup["cogDataConfig"][elem]["wordEmbSpecifics"][name] = parameters

    with open(output_path / configFile,'w') as fW:
        json.dump(setup,fW,indent=4,sort_keys = True)

if __name__ == '__main__':
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
    editor_config_json(embedding_parameters)