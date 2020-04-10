import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

def chunk(input_path_name,
          output_path_name,
          input_file_name,
          output_file_name,
          number_of_chunks=4,
          truncate_first_line=False):
    '''
    Splits embeddings into the given number of chunks.

    :param input_path_name: Input directory of embeddings
    :param output_path_name: Output directory for chunked embeddings
    :param input_file_name: File name of input embedidngs
    :param output_file_name: Base name of output chunks
    '''
    input_path = Path(input_path_name)
    output_path = Path(output_path_name)
    
    if truncate_first_line:
        skip_lines = 1
    else:
        skip_lines = 0
    
    rows = 0
    with open(input_path / input_file_name) as f:
        for _ in f:
            rows += 1
        
    chunk_size = rows // number_of_chunks
    rest = rows % number_of_chunks

    with open(input_path / input_file_name) as f:
        # Forward lines
        for i in range(skip_lines):
            next(f)
        
        # Iterate over chunks
        for i in range(0, number_of_chunks):
            if i == number_of_chunks - 1:
                csize = csize + rest
            else:
                csize = chunk_size

            with open(output_path / "{}_{}.txt".format(output_file_name,
                                            str(i)),
                                            "w") as f_out:
                print('Chunk {}/{}:'.format(i+1, number_of_chunks))
                for _ in tqdm(list(range(0, chunk_size))):
                    try:
                        f_out.write(next(f))
                    except StopIteration:
                        break
                print()


def update(left_df, right_df, on_column, columns_to_omit, whole_row):
    '''
    Performs a left-merge between two dataframes on the given column.
    
    :param left_df: Left-hand df (which is updated)
    :param right_df: Right-hand df (providing the values considered for updates)
    :param on_column: Column on which to merge
    :param columns_to_omit: Columns which should be omitted
    :param whole_row: When encountering a NaN occurrence, whether to update
                      the whole row (True) or just on Nan values (False).
    '''
    # Both dataframes have to have same column names
    header = list(left_df)
    header = header[columns_to_omit:]
    start = left_df.shape[1]
    to_update = left_df.merge(right_df, on=on_column, how='left').iloc[:, start:].dropna()
    to_update.columns = header

    if(whole_row):
        # UPDATE whole row when NaN appears
        left_df.loc[left_df[header[0]].isnull(), header] = to_update
    else:
        # UPDATE just on NaN values
        for elem in header:
            left_df.loc[left_df[elem].isnull(),elem] = to_update[elem]

    return left_df


def df_multi_join(df_cognitive_data, df_word_embedding, chunk_number=4):
    '''
    Join cognitive data and word embeddings DataFrames via chunks,
    to perform 'MemorySafe'-join.
    
    :param df_cognitive_data: DataFrame containing cognitive data
    :param df_word_embedding: DataFrame containing word embeddings
    :param chunk_number: Number of chunks (default: 4)
    '''
    df_join = df_cognitive_data
    rows = df_word_embedding.shape[0]
    chunk_size = rows // chunk_number
    rest = rows % chunk_number
    for i in range(0, chunk_number):
        begin = chunk_size * i
        end = chunk_size * (i + 1)
        if i == 0:
            df_join = pd.merge(df_join, df_word_embedding.iloc[begin:end, :], how='left', on=['word'])
        else:
            if i == chunk_number - 1:
                end = end + rest
            update(df_join, df_word_embedding.iloc[begin:end, :], on_column=['word'], columns_to_omit=df_cognitive_data.shape[1], whole_row=True)

    return df_join


def multi_join(mode, config, emb_type, df_cognitive_data, word_embedding):
    '''
    Joins a cognitive data DataFrame with chunked embeddings. Embeddings
    are expected to be provided as space-separated CSVs. Path of the CSVs 
    is specified by the configuration dictionary.

    :param config: configuration dictionary specifying embedding parameters
                   (chunk numbers, file name, ending)
    :param emb_type: 'sentence' or 'word' for corresponding embeddings
    :param df_cognitive_data: Cognitive data DataFrame
    :param word_embedding: word embedding name (required for lookup in
                           configuration dictionary)
    '''
    if mode == 'proper':
        emb_key = 'wordEmbConfig'
    else:
        emb_key = 'randEmbConfig'

    emb_type = config['type']
        
    # READ Datasets into dataframes

    # Join from chunked FILE
    df_join = df_cognitive_data
    word_emb_prop = config[emb_key][word_embedding]
    chunk_number = word_emb_prop["chunk_number"]
    base_path = Path(config['PATH'])
    embedding_path = word_emb_prop["path"].rsplit('/', maxsplit=1)[0]
    path = base_path / embedding_path
    chunked_file = word_emb_prop["chunked_file"]
    ending = word_emb_prop["chunk_ending"]

    for i in range(0, chunk_number):
        with open(path  / '{}_{}{}'.format(chunked_file, str(i), ending)) as f:
            first_line = next(f)
        dimensions = len(first_line.split(" ")) - 1

        if emb_type == 'word':
            df = pd.read_csv(path  / '{}_{}{}'.format(chunked_file, str(i), ending),
                             sep=" ",
                             encoding="utf-8",
                             quoting=csv.QUOTE_NONE,
                             names=[emb_type] + ['x_{}'.format(idx + 1) for idx in range(dimensions)])
        elif emb_type == 'sentence':
            df = pd.read_csv(path  / '{}_{}{}'.format(chunked_file, str(i), ending),
                             sep=" ",
                             encoding="utf-8",
                             quotechar='"',
                             quoting=csv.QUOTE_NONNUMERIC,
                             doublequote=True,
                             names=[emb_type] + ['x_{}'.format(idx + 1) for idx in range(dimensions)])

        if i == 0:
            df_join = pd.merge(df_join, df, how='left', on=[emb_type])
        else:
            update(df_join, df, on_column=[emb_type], columns_to_omit=df_cognitive_data.shape[1], whole_row=True)

    return df_join


def split_folds(strings, X, y, folds, seed, balance, sub_sources):
    '''
    Splits the given data (strings, vecotrs and labels) into the given number of
    folds. Splitting is deterministic, as per the given random seed.

    :param strings: np.array with strings, same order, corresponding to X and y vectors
    :param X: np.array with data
    :param y: np.array with labels
    :param folds: number of folds
    :return: X_train = [trainingset1, trainingset2, trainingset3,...]
    '''

    np.random.seed(seed)
    np.random.shuffle(strings)

    np.random.seed(seed)
    np.random.shuffle(X)

    np.random.seed(seed)
    np.random.shuffle(y)

    # Apply stratified K-Folds if composite source, to preserve the percentage of each subsource
    if sub_sources is not None:
        np.random.seed(seed)
        np.random.shuffle(sub_sources)

        selector = StratifiedKFold(n_splits=folds, shuffle=False, random_state=None)
        #print(selector.get_n_splits(X, sub_sources))
        selector_splits = selector.split(X, sub_sources)
    else:
        selector = KFold(n_splits=folds, shuffle=False, random_state=None)
        #print(selector.get_n_splits(X))
        selector_splits = selector.split(X)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    strings_test = []
    sub_sources_train = []

    for train_index, test_index in selector_splits:
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
        if sub_sources is not None:
            sub_sources_train.append(sub_sources[train_index])
        strings_test.append(strings[test_index])

    # Balance (only!) training data, if applicable
    if (sub_sources is not None) and balance:
        X_train_resampled = []
        y_train_resampled = []

        for idx, (sub_sources_fold, X_train_fold, y_train_fold) in enumerate(zip(sub_sources_train, X_train, y_train)):
            
            print("Fold {} length prior to resampling: {} / Distribution of sources: {}".format(idx + 1, len(X_train), collections.Counter(sub_sources_fold)))
            
            X_y_train_fold = list(zip(X_train_fold, y_train_fold))
            ros = RandomOverSampler()
            X_y_train_fold_resampled, sub_sources_resampled = ros.fit_resample(X_y_train_fold, sub_sources_fold)
            X_y_train_fold_resampled = [tuple(x_y) for x_y in X_y_train_fold_resampled]
            X_train_fold, y_train_fold = list(zip(*X_y_train_fold_resampled))
            
            print("Fold {} length after resampling: {} / Distribution of sources: {}".format(idx + 1, len(X_train), collections.Counter(sub_sources_resampled)))

            X_train_resampled.append(np.vstack(X_train_fold))
            y_train_resampled.append(np.vstack(y_train_fold))

        X_train, y_train = X_train_resampled, y_train_resampled
    
    return strings_test, X_train, y_train, X_test, y_test


def data_handler(mode, config, stratified_sampling, balance, word_embedding, cognitive_data, feature, truncate_first_line):
    '''
    Loads and merges specified cognitive data and word embeddings through
    given configuraiton dictionary. Returns chunked data and labels for
    k-fold cross-validation.

    :param mode: Type of embeddings, either 'proper' or 'random'
    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    :param truncate_first_line: If the first line of the embedding file should be truncated (when containing meta data)
    '''
    if mode == 'proper':
        emb_key = 'wordEmbConfig'
    else:
        emb_key = 'randEmbConfig'

    emb_type = config['type']
    # READ Datasets into dataframes
    if emb_type == 'word':
        df_cognitive_data = pd.read_csv(Path(config['PATH']) / config['cogDataConfig'][cognitive_data]['dataset'],
                                        sep=" ",
                                        encoding="utf-8",
                                        quotechar=None,
                                        quoting=csv.QUOTE_NONE,
                                        doublequote=False)
    elif emb_type == 'sentence':
        df_cognitive_data = pd.read_csv(Path(config['PATH']) / config['cogDataConfig'][cognitive_data]['dataset'],
                                        sep=" ",
                                        encoding="utf-8",
                                        quotechar='"',
                                        quoting=csv.QUOTE_NONNUMERIC,
                                        doublequote=True)

    # In case it's a single output cogData we just need the single feature
    if config['cogDataConfig'][cognitive_data]['type'] == "single_output":
        df_cognitive_data = df_cognitive_data[[emb_type, feature]]
    df_cognitive_data.dropna(inplace=True)

    if (config[emb_key][word_embedding]["chunked"]):
        df_join = multi_join(mode, config, emb_type, df_cognitive_data, word_embedding)
    else:
        if truncate_first_line:
            skip_rows = 0
        else:
            skip_rows = None

        dimensions = config[emb_key][word_embedding]["dimensions"]
        
        if emb_type == 'word':
            df_word_embedding = pd.read_csv(Path(config['PATH']) / config[emb_key][word_embedding]["path"],
                                            sep=" ",
                                            encoding="utf-8",
                                            quoting=csv.QUOTE_NONE,
                                            skiprows=skip_rows,
                                            names=[emb_type] + ['x_{}'.format(idx + 1) for idx in range(dimensions)])
        elif emb_type == 'sentence':
            df_word_embedding = pd.read_csv(Path(config['PATH']) / config[emb_key][word_embedding]["path"],
                                            sep=" ",
                                            encoding="utf-8",
                                            quotechar='"',
                                            quoting=csv.QUOTE_NONNUMERIC,
                                            doublequote=True,
                                            names=[emb_type] + ['x_{}'.format(idx + 1) for idx in range(dimensions)])

        # Left (outer) Join to get wordembedding vectors for all strings in cognitive dataset
        df_join = pd.merge(df_cognitive_data, df_word_embedding, how='left', on=[emb_type])

    df_join.dropna(inplace=True)
    
    if stratified_sampling:
        df_join['source'] = df_join['source'].astype(int)
        for source, count in df_join['source'].value_counts().to_dict().items():
            if count < config['folds']:
                #print("Source {} has too few elements, dropping ...".format(source))
                df_join = df_join[df_join['source'] != source]

        sub_sources = df_join['source'].to_numpy()
    else:
        sub_sources = None
    
    df_join.drop(columns=['source'], inplace=True, errors='ignore')
    df_cognitive_data.drop(columns=['source'], inplace=True, errors='ignore')

    strings = df_join[emb_type]
    strings = np.array(strings, dtype='str').reshape(-1,1)

    df_join.drop([emb_type], axis=1, inplace=True)

    if config['cogDataConfig'][cognitive_data]['type'] == "single_output":
        y = df_join[feature]
        y = np.array(y, dtype='float').reshape(-1, 1)

        X = df_join.drop(feature, axis=1)
        X = np.array(X, dtype='float')
    else:
        features = df_cognitive_data.columns[1:]
        y = df_join[features]
        y = np.array(y, dtype='float')

        X = df_join.drop(features, axis=1)
        X = np.array(X, dtype='float')

    return split_folds(strings, X, y, config["folds"], config["seed"], balance, sub_sources)
