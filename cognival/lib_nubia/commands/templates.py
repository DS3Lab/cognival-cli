WORD_EMB_CONFIG_FIELDS = set(["dimensions",
                              "chunk_number",
                              "chunked",
                              "chunked_file",
                              "chunk_ending",
                              "path",
                              "truncate_first_line",
                              "random_embedding"])

MAIN_CONFIG_TEMPLATE = {
                        "type": None,
                        "PATH": None,
                        "cogDataConfig": {},
                        "n_proc": None,
                        "folds": None,
                        "outputDir": None,
                        "seed": None,
                        "run_id": None,
                        "wordEmbConfig": {},
                        "randEmbConfig": {},
                        "randEmbSetToParts": {}
                        }

COGNITIVE_CONFIG_TEMPLATE = {
                            "dataset": None,
                            "parent": None,
                            "multi_hypothesis": None,
                            "multi_file": None,
                            'kpca_n_dim': None,
                            'kpca_gamma': None,
                            'kpca_kernel': None,
                            "stratified_sampling": None,
                            "balance": None,
                            "modality": None,
                            "features": [],
                            "type": None,
                            "wordEmbSpecifics": {}
                            }

EMBEDDING_PARAMET_TEMPLATE = {
                            "activations": [],
                            "batch_size": [],
                            "cv_split": None,
                            "epochs": [],
                            "layers": [],
                            "validation_split": None
                            }

EMBEDDING_CONFIG_TEMPLATE = {
                            "dimensions": None,
                            "chunk_number": 0,
                            "chunked": 0,
                            "chunked_file": None,
                            "chunk_ending": None,
                            "path": None,
                            "truncate_first_line":False,
                            "random_embedding":None
                            }
