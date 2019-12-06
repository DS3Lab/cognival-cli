from pathlib import Path

import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold

def chunk(input_path_name,
          output_path_name,
          input_file_name,
          output_file_name,
          number_of_chunks=4):
    '''
    Splits embeddings into the given number of chunks.

    :param input_path_name: Input directory of embeddings
    :param output_path_name: Output directory for chunked embeddings
    :param input_file_name: File name of input embedidngs
    :param output_file_name: Base name of output chunks
    '''
    input_path = Path(input_path_name)
    output_path = Path(output_path_name)
    df_wE = pd.read_csv(input_path / input_file_name, sep=" ",
                        encoding="utf-8", quoting=csv.QUOTE_NONE)
    rows = df_wE.shape[0]
    chunk_size = rows // number_of_chunks
    rest = rows % number_of_chunks
    for i in range(0, number_of_chunks):
        begin = chunk_size * i
        end = chunk_size * (i + 1)
        if i == number_of_chunks - 1:
            end = end + rest
        df = df_wE.iloc[begin:end,:]
        df.to_csv(output_path / "{}_{}.txt".format(output_file_name, str(i)), sep=" ", encoding="utf-8")


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


def multi_join(config, df_cognitive_data, word_embedding):
    '''
    Joins a cognitive data DataFrame with chunked embeddings. Embeddings
    are expected to be provided as space-separated CSVs. Path of the CSVs 
    is specified by the configuration dictionary.

    :param config: configuration dictionary specifying embedding parameters
                   (chunk numbers, file name, ending)
    :param df_cognitive_data: Cognitive data DataFrame
    :param word_embedding: word embedding name (required for lookup in
                           configuration dictionary)
    '''
    # Join from chunked FILE
    df_join = df_cognitive_data
    chunk_number = config['wordEmbConfig'][word_embedding]["chunk_number"]
    file_name = config['PATH'] + config['wordEmbConfig'][word_embedding]["chunked_file"]
    ending = config['wordEmbConfig'][word_embedding]["ending"]

    for i in range(0, chunk_number):
        df = pd.read_csv(file_name + str(i) + ending, sep=" ",
                         encoding="utf-8", quoting=csv.QUOTE_NONE)
        df.drop(df.columns[0], axis=1, inplace=True)
        if i == 0:
            df_join = pd.merge(df_join, df, how='left', on=['word'])
        else:
            update(df_join, df, on_column=['word'], columns_to_omit=df_cognitive_data.shape[1], whole_row=True)

    return df_join


def split_folds(words, X, y, folds, seed):
    '''
    Splits the given data (words, vecotrs and labels) into the given number of
    folds. Splitting is deterministic, as per the given random seed.

    :param words: np.array with words, same order, corresponding to X and y vectors
    :param X: np.array with data
    :param y: np.array with labels
    :param folds: number of folds
    :return: X_train = [trainingset1, trainingset2, trainingset3,...]
    '''

    np.random.seed(seed)
    np.random.shuffle(words)

    np.random.seed(seed)
    np.random.shuffle(X)

    np.random.seed(seed)
    np.random.shuffle(y)


    kf = KFold(n_splits=folds, shuffle=False, random_state=None)
    kf.get_n_splits(X)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    words_test = []

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
        words_test.append(words[test_index])

    return words_test, X_train, y_train, X_test, y_test


def data_handler(config, word_embedding, cognitive_data, feature):
    '''
    Loads and merges specified cognitive data and word embeddings through
    given configuraiton dictionary. Returns chunked data and labels for
    k-fold cross-validation.

    :param config: Configuration dictionary
    :param word_embedding: String specifying word embedding (configuration key)
    :param cognitive_data: String specifying cognitiv data source (configuration key)
    :param feature: Cognitive data feature to be predicted
    '''
    # READ Datasets into dataframes
    df_cognitive_data = pd.read_csv(config['PATH'] + config['cogDataConfig'][cognitive_data]['dataset'], sep=" ")

    # In case it's a single output cogData we just need the single feature
    if config['cogDataConfig'][cognitive_data]['type'] == "single_output":
        df_cognitive_data = df_cognitive_data[['word',feature]]
    df_cognitive_data.dropna(inplace=True)

    if (config['wordEmbConfig'][word_embedding]["chunked"]):
        df_join = multi_join(config, df_cognitive_data, word_embedding)
    else:
        df_word_embedding = pd.read_csv(config['PATH'] + config['wordEmbConfig'][word_embedding]["path"], sep=" ",
                            encoding="utf-8", quoting=csv.QUOTE_NONE)
        # Left (outer) Join to get wordembedding vectors for all words in cognitive dataset
        df_join = pd.merge(df_cognitive_data, df_word_embedding, how='left', on=['word'])

    df_join.dropna(inplace=True)

    words = df_join['word']
    words = np.array(words, dtype='str').reshape(-1,1)

    df_join.drop(['word'], axis=1, inplace=True)

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

    return split_folds(words, X, y, config["folds"], config["seed"] )


def main():
    import json
    
    with open('../config/example_1.json', 'r') as fr:
        config = json.load(fr)
    
    we = 'glove-50'
    feat = 'ALL_DIM'
    cds = []
    
    cds = ['zuco-eeg']
    
    for cd in cds:
        words_test, X_train, y_train, X_test, y_test = data_handler(config, we, cd, feat)        
        print("SUCCESS: " + cd)

if __name__=="__main__":
    main()