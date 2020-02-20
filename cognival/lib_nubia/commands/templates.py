WORD_EMB_CONFIG_FIELDS = set(["chunk_number",
                              "chunked",
                              "chunked_file",
                              "chunk_ending",
                              "path",
                              "truncate_first_line",
                              "random_embedding"])

MAIN_CONFIG_TEMPLATE = {
                        "PATH": None,
                        "cogDataConfig": {},
                        "n_proc": None,
                        "folds": None,
                        "outputDir": None,
                        "seed": None,
                        "version": None,
                        "wordEmbConfig": {},
                        "randEmbConfig": {},
                        "randEmbSetToParts": {}
                        }

COGNITIVE_CONFIG_TEMPLATE = {
                            "dataset": None,
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
                            "chunk_number": 0,
                            "chunked": 0,
                            "chunked_file": None,
                            "ending": None,
                            "path": None
                            }
