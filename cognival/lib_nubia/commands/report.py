import csv
import datetime
import collections
import json
import webbrowser
import base64
import os
from copy import deepcopy
from io import BytesIO
from operator import truediv
from pathlib import Path

import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from jinja2 import Environment, PackageLoader, select_autoescape
import pdfkit
from operator import truediv as div
from subprocess import Popen, DEVNULL

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LinearLocator, FormatStrFormatter, MultipleLocator
from natsort import index_natsorted, order_by_index, ns
import seaborn as sns
import random

from termcolor import cprint
from .utils import _open_config

bar_colors = ['#222222', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'] + [v for k, v in sns.xkcd_rgb.items() if not any([prop in k for prop in ['faded', 'light', 'pale', 'dark']])]

MODALITIES_SHORT_TO_FULL = {'eeg':'EEG',
                            'eye-tracking': 'Eye-Tracking',
                            'fmri': 'fMRI'}

def load_jinja_template(path=['reporting', 'templates'], template_file='cognival_report.html'):
    env = Environment(
        loader=PackageLoader(*path),
        autoescape=select_autoescape(['html', 'xml'])
    )

    env.globals.update(sig_status=sig_status,
                       get_sig=get_sig)
    template = env.get_template(template_file)
    return template

def sig_status(sig_value):
    try:
        sig_ratio = truediv(*map(int, sig_value.split("/")))
        if sig_ratio == 0.0:
            return 'none'
        elif sig_ratio == 1.0:
            return 'all'
        else:
            return 'some'
    except AttributeError:
        if sig_value:
            return 'all'
        else:
            return 'none'


def get_sig(sig_value, average_multi_hypothesis):
    if isinstance(sig_value, str) or not average_multi_hypothesis:
        return sig_value
    elif sig_value:
        return '1/1'
    else:
        return '0/1'

# Source: https://stackoverflow.com/a/53218939
def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


def sig_bar_plot(df):
    '''
    Generates a bar plot with average MSE and significance stats per embedding type.
    '''
    df['Embeddings'] = df['Embeddings'].apply(lambda x: {\
                'fasttext-wiki-subword':'fastText',
                'fasttext-cc':'fastText',
                'fasttext-cc-subword':'fastText',
                'bert-base-cased': 'BERT',
                'bert-large-cased': 'BERT',
                'skip-thoughts-bi': 'Skip-Thought',
                'elmo-sentence': 'ELMo',
                'elmo-sentence-large': 'ELMo',
                'glove.6B.50': 'GloVe',
                'use': 'USE',
                'infersent': 'InferSent',
                'power-mean': 'Power-Mean'}.get(x, x))
    df = unnesting(df, ["MSE CV folds"])
    df['Embeddings'] = pd.Categorical(df['Embeddings'], ["GloVe", "fastText", "ELMo", "Power-Mean", "Skip-Thought", "BERT", "InferSent", "USE"])
    df.sort_values(['Embeddings', 'Type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    max_y = max(df["MSE CV folds"])
    min_y = min(df["MSE CV folds"])
    df = df[sorted(df.columns)]
    # Sort embedding names naturally
    df = df.reindex(index=order_by_index(df.index, index_natsorted(df['Embeddings'], alg=ns.IC)))
    sig_labels = [row['Significance'] for _, row in df.iterrows() if row['Significance'] != '-']
    # Make the barplot
    num_embeddings = len(df['Embeddings'].unique())
     # Set style
    sns.set(style="whitegrid", color_codes=True)
    fig = plt.figure()
    bar = sns.boxplot(x="Embeddings", y="MSE CV folds", hue="Type", data=df[df.columns.difference(['Significance', 'Modality'])], showfliers=False)
    bar.get_legend().remove()
    #bar.set(ylim=(min_y - 0.1*min_y, max_y + 0.1*max_y))

    # Loop over the bars
    import matplotlib.patches
    patches = bar.findobj(matplotlib.patches.PathPatch)

    # proper embeddings (get xkcd colors)
    for idx, thisbar in enumerate(patches[::2]):
        #thisbar.set_width(0.95 * thisbar.get_width())
        thisbar.set_facecolor(bar_colors[idx])
        #x = thisbar.get_x()
        #y = -0.015*max_y
        #bar.annotate('({})\n{:.2e}'.format(sig_labels[idx], thisbar.get_height()), (x, y), rotation=45, ha='left')   

    # random embeddings (grey)
    for idx, thisbar in enumerate(patches[1::2]):
        #thisbar.set_width(0.95 * thisbar.get_width())
        thisbar.set_facecolor("#dddddd")
        #x = thisbar.get_x()
        #y = -0.01*max_y
        #bar.annotate('{:.2e}'.format(thisbar.get_height()), (x, y), rotation=45, ha='center') 
    
    # Adjust the margins
    plt.subplots_adjust(bottom= 0.2, top = 0.8)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    bar.set_yticklabels(['']*len(bar.get_yticklabels()))
    y_major = MultipleLocator(10**(np.ceil(np.log10(min_y) - 1)))
    y_minor = MultipleLocator(10**(np.ceil(np.log10(min_y) - 1)))

    bar.yaxis.set_major_locator(y_major)
    bar.yaxis.set_minor_locator(y_minor)

    sf = ScalarFormatter()
    sf.set_scientific(False)
    bar.yaxis.set_major_formatter(sf)
    #bar.yaxis.set_minor_formatter(sf)
    plt.draw()
    plt.grid(which='both', axis='y')
    # Scrub minor labels
    ytl = [item.get_text() for item in bar.get_yticklabels(minor=True)]
    ytl = len(ytl)*[""]
    bar.set_yticklabels(ytl, minor=True)
    ytl = [item.get_text() for item in bar.get_yticklabels()]
    # Scrub major labels
    ytl_start = ytl[:2]
    prev = None
    ytl_idx = set([idx**2 if idx < 5 else (idx-2)**3 for idx, _ in enumerate(ytl[2:])])
    for idx, x in enumerate(ytl[2:]):
        if idx in ytl_idx:
            ytl_start.append(x)
        else:
            ytl_start.append('')
    ytl = ytl_start
    bar.set_yticklabels(ytl)
    bar.xaxis.label.set_visible(False)
    with sns.plotting_context("notebook", rc={"font.size":16,"axes.titlesize":24,"axes.labelsize":22}, font_scale=4):
        bar.set_ylabel(bar.get_ylabel(), fontsize=18)
        bar.tick_params(labelsize=16)
        with BytesIO() as figfile:
            fig.set_size_inches(int(num_embeddings*0.75), 6)
            plt.savefig(figfile, format='png', dpi=300, bbox_inches="tight")
            figfile.seek(0)  # rewind to beginning of file
            statsfig_b64 = base64.b64encode(figfile.getvalue()).decode('utf8')
        return statsfig_b64

def agg_stats_over_time_plots(agg_reports_dict, run_id):
    '''
    Generates line plots with aggregate stats over time (run_ids)
    Ø MSE Baseline, Ø MSE Embeddings, Significance
    '''
    modality_to_plots = {}
    for modality, run_ids in agg_reports_dict.items():
        df_list = []
        for agg_run_id, agg_params in run_ids.items():
            df = pd.DataFrame.from_dict(agg_params)
            df.reset_index(inplace=True)
            df.rename(columns={'index':'Embeddings'}, inplace=True)
            df['run_id'] = [agg_run_id]*len(df)
            df['Significance'] = df['Significance'].apply(lambda x: div(*map(int, x.split('/'))))
            df_list.append(df)
        df = pd.concat(df_list)
        df = df[df['run_id'] <= run_id]

        # Skip if modality is not evaluated in current run or only one run_id
        if not run_id in df['run_id'].values or len(df['run_id'].values) == 1:
            continue

        plots_b64 = []

        for measure in ["ØØ MSE Baseline", "ØØ MSE Embeddings", "Significance"]:
            plt.clf()
            plt.cla()
            plt.figure()
            
            try:
                df_sub = df[['Embeddings', measure, 'run_id']]
            except KeyError:
                continue
            df_sub_list = []

            for emb in df_sub['Embeddings'].unique():
                df_subsub = df_sub[df_sub['Embeddings'] == emb].copy()
                df_subsub.rename(columns={measure:emb}, inplace=True)
                df_subsub = df_subsub[[emb, 'run_id']]
                df_sub_list.append(pd.melt(df_subsub, ['run_id']))
            df_sub = pd.concat(df_sub_list, axis=0)
            df_sub.reset_index(inplace=True, drop=True)

            plot = sns.lineplot(x='run_id', y='value', hue='variable', data=df_sub)
            plot.set_title(measure)

            plot.locator_params(integer=True)
            with BytesIO() as figfile:
                plot.figure.set_size_inches(8, 4)
                plot.legend(title='Embeddings', loc='upper right', bbox_to_anchor=(1.3, 1), shadow=True, ncol=1)
                # TODO: Change this hack if possible (see https://stackoverflow.com/a/54537872)
                #Hack to remove the first legend entry (which is the undesired title)
                vpacker = plot.get_legend()._legend_handle_box.get_children()[0]
                vpacker._children = vpacker.get_children()[1:]
                plot.figure.savefig(figfile,
                            format='png',
                            dpi=300,
                            bbox_inches="tight")
                figfile.seek(0)  # rewind to beginning of file
                statsfig_b64 = base64.b64encode(figfile.getvalue()).decode('utf8')
                plots_b64.append(statsfig_b64)
        modality_to_plots[MODALITIES_SHORT_TO_FULL[modality]] = plots_b64
    return modality_to_plots


def generate_report(configuration,
                    test,
                    run_id,
                    resources_path,
                    precision,
                    average_multi_hypothesis,
                    train_history_plots,
                    features,
                    err_tables,
                    err_tables_sample_n,
                    err_tables_discard_na,
                    export_err_tables=False,
                    html=True,
                    pdf=False,
                    open_html=False,
                    open_pdf=False):
    '''
    Generates report from significance test results and aggregated statistics for given configuration
    and configuration run_id.
    '''
    template = load_jinja_template()

    config_dict = _open_config(configuration, resources_path, quiet=True, quiet_errors=True)
    if not config_dict:
        return
    cprint('Generating CogniVal report ...', 'green')

    config_dict_report = {k:v for k,v in config_dict.items() if k not in ('cogDataConfig',
                                                                          'wordEmbConfig',
                                                                          'randEmbConfig',
                                                                          'randEmbSetToParts',
                                                                          'run_id')}

    # Get mapping of previous run_id (current not yet executed)
    if not run_id:
        run_id = config_dict['run_id'] - 1
    elif run_id >= config_dict['run_id']:
        cprint('Run ID {} exceeds last run_id for which results were generated ({}), aborting ...'.format(run_id, config_dict['run_id'] - 1), 'red')
        return
    if not run_id:
        cprint('No experimental runs performed yet for configuration {}, aborting ...'.format(configuration), 'red')
        return

    config_dict_report['run_id'] = run_id
    config_dict_report['sig_test'] = test

    if not html and not pdf:
        cprint('No output format enabled, aborting ...', 'red')
        return

    out_dir = Path(config_dict['PATH']) / config_dict['outputDir']

    with open(out_dir / 'mapping_{}.json'.format(run_id)) as f:
        mapping_dict = json.load(f)

    report_dir = out_dir / 'reports'

    experiment_to_path = {}
    training_history_plots = {}
    random_to_proper = {}
    avg_error_single_dfs = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    avg_error_dfs = collections.defaultdict(lambda: collections.defaultdict(dict))

    experiments_dir = out_dir / 'experiments'
    avg_errors_dir = out_dir / 'average_errors'
    report_dir = out_dir / 'reports'
    sig_test_reports_dict = collections.defaultdict(dict)
    agg_reports_dict = collections.defaultdict(dict)
    results = []

    for key, value in mapping_dict.items():
        experiment_to_path[key] = experiments_dir / value['proper'] / '{}.json'.format(value['embedding'])
        try:
            random_to_proper[value['random_name']] = key
        except KeyError:
            pass

        try:
            if train_history_plots:
                with open(experiments_dir / value['proper'] / '{}.png'.format(value['embedding']), 'rb') as f:
                    figdata_b64 = base64.b64encode(f.read()).decode('utf8')
                    training_history_plots[key] = figdata_b64
            else:
                training_history_plots = None
        except FileNotFoundError:
            continue

    if err_tables:
        # Collection avg error results and generating min-max scaled tables
        for path, _, error_csvs in os.walk(avg_errors_dir):
            if error_csvs:
                for error_csv in error_csvs:
                    if not 'baseline' in error_csv:
                        modality, ver = path.split('/')[-2:]
                        ver = int(ver)
                        if ver != run_id:
                            continue
                        df = pd.read_csv(Path(path) / error_csv,
                                 sep=" ",
                                 encoding="utf-8",
                                 quotechar='"',
                                 quoting=csv.QUOTE_NONNUMERIC,
                                 doublequote=True)
                        experiment = error_csv.rsplit('.', maxsplit=1)[0].replace('embeddings_avg_errors_', '') 
                        
                        with open(experiment_to_path[experiment]) as f:
                            result_dict = json.load(f)
                        
                        source, feature, emb = result_dict['cognitiveData'], result_dict['feature'], result_dict['wordEmbedding']
                        
                        feature = feature if feature != 'ALL_DIM' else '—'
                        df.rename(columns={'error':emb}, inplace=True)
                        df.set_index(config_dict['type'], inplace=True)
                        assert not df.index.has_duplicates
                        avg_error_single_dfs[modality][source][feature][emb] = df

        print("Creating heatmap (error) tables ...")
        
        with tqdm() as pbar:
            for modality, modality_dict in avg_error_single_dfs.items():
                for source, source_dict in modality_dict.items():
                    for feature, feature_dict in source_dict.items():
                        pbar.update()
                        embeddings, dfs = zip(*list(feature_dict.items()))
                        df = pd.concat(dfs, axis='columns')
                        df -= df.min().min()
                        df /= df.max().max()
                        df.reset_index(inplace=True)
                        df.rename(columns={'index': config_dict['type']}, inplace=True)

                        if export_err_tables:
                            err_t_path = report_dir / modality / str(run_id) / 'error_tables'
                            if feature == '—':
                                err_t_file = '{}_error_table.parquet.gz'.format(source)
                            else:
                                err_t_file = '{}_{}_error_table.parquet.gz'.format(source,
                                                                                   feature)
                            
                            os.makedirs(err_t_path, exist_ok=True)
                            df.to_parquet(err_t_path / err_t_file,
                                          engine='auto',
                                          compression='gzip')

                        if err_tables_discard_na:
                            df.dropna(inplace=True)

                        if err_tables_sample_n:
                            df = df.sample(n=err_tables_sample_n)

                        avg_error_dfs[modality.upper()][(source, feature)] = df

    # Collecting significance test results and aggregation results
    for path, _, reports in os.walk(report_dir):
        if reports:
            for report in reports:
                if report.endswith('json'):
                    modality, ver = path.split('/')[-2:]
                    with open(Path(path) / report) as f_sig:
                        report_dict = json.loads(f_sig.read())
                    if report == '{}.json'.format(test):
                        sig_test_reports_dict[modality][int(ver)] = report_dict
                    elif report == 'aggregated_scores.json':
                        agg_reports_dict[modality][int(ver)] = report_dict


    # Detail (proper)
    # If significance tests have been performed
    if sig_test_reports_dict:
        for modality, mod_report_run_ids in sig_test_reports_dict.items():
            mod_report = mod_report_run_ids.get(run_id, None)
            if not mod_report:
                continue

            for experiment, sig_test_result in mod_report['hypotheses'].items():
                with open(experiment_to_path[experiment]) as f:
                    result_dict = json.load(f)
                result = {'Modality': MODALITIES_SHORT_TO_FULL[modality],
                        'Ø MSE': result_dict['AVERAGE_MSE'],
                        'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                        'Word embedding': result_dict['wordEmbedding'],
                        'Subject': result_dict['cognitiveData'] if result_dict['multi_hypothesis'] == 'subject' else '-',
                        'Cognitive source': result_dict['cognitiveParent'],
                        'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature'],
                        'Details': result_dict['details'],
                        **sig_test_result}
                results.append(result)
    # If no significance tests have been performed
    if not results:
        for experiment, result_json_path in experiment_to_path.items():
            with open(result_json_path) as f:
                result_dict = json.load(f)
                result = {'Modality': MODALITIES_SHORT_TO_FULL[result_dict['modality']],
                        'Ø MSE': result_dict['AVERAGE_MSE'],
                        'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                        'Word embedding': result_dict['wordEmbedding'],
                        'Subject': result_dict['cognitiveData'] if result_dict['multi_hypothesis'] == 'subject' else '-',
                        'Cognitive source': result_dict['cognitiveParent'],
                        'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature'],
                        'Details': result_dict['details']}
                results.append(result)

    df_details = pd.DataFrame(results)
    
    if 'alpha' in df_details:
        df_details.rename(columns={'alpha': 'α',
                                   'bonferroni_alpha': 'Bonferroni α',
                                   'p_value': 'p'}, inplace=True)
        numtypes = ['Ø MSE', 'SD MSE', 'α', 'Bonferroni α', 'p']
    else:
        numtypes = ['Ø MSE', 'SD MSE']

    df_details = df_details.astype({k:float for k in numtypes})
    
    if average_multi_hypothesis:
        group_by_keys = ['Modality', 'Word embedding', 'Cognitive source', 'Details']
        df_details_atomic = df_details.copy()[(df_details['Subject'] == '-') & (df_details['Feature'] == '-')]
        df_details_feat_avg = df_details.copy()[(df_details['Subject'] == '-') & (df_details['Feature'] != '-')]
        df_details_multi_subj_avg = df_details.copy()[df_details['Subject'] != '-']
        df_details_atomic = df_details_atomic.drop(['Subject'], axis='columns')
        df_details_feat_avg = df_details_feat_avg.drop(['Subject'], axis='columns')
        df_details_atomic = df_details_atomic.groupby(group_by_keys).mean()
        df_details_atomic['Hypotheses'] = 1
        df_details_atomic['Features'] = '-'
        agg_feat_avg_dict = {'Feature': lambda col: ', '.join(col),
                    **{k:'mean' for k in numtypes}}
        if 'significant' in df_details:
            agg_feat_avg_dict['significant'] = lambda col: '{}/{}'.format(list(col).count(True), len(col))
        df_details_feat_avg = df_details_feat_avg.groupby(group_by_keys) \
                                                 .agg(agg_feat_avg_dict) \
                                                 .rename(columns={'Feature': 'Features'})
        df_details_feat_avg['Hypotheses'] = df_details_feat_avg['Features'].apply(lambda x: len(x.split(', ')))

        agg_multi_subj_dict = {'Subject': 'size',
                               **{k:'mean' for k in numtypes}}
        if 'significant' in df_details:
            agg_multi_subj_dict['significant'] = lambda col: '{}/{}'.format(list(col).count(True), len(col))

        df_details_multi_subj_avg = df_details_multi_subj_avg.groupby(group_by_keys) \
                                                             .agg(agg_multi_subj_dict) \
                                                             .rename(columns={'Subject': 'Hypotheses'})
        df_details_multi_subj_avg['Features'] = '-'
        df_details = pd.concat([df_details_atomic, df_details_feat_avg, df_details_multi_subj_avg])

        df_details['Hypotheses'] = df_details['Hypotheses'].astype(int)
        df_details.reset_index(inplace=True)
    for col in ['Ø MSE', 'SD MSE', 'p', 'α']:
        try:
            df_details[col] = df_details[col].map(lambda x: ('{:.%df}' % precision).format(x))
        except KeyError:
            pass

    if 'significant' in df_details:
        df_details['Bonferroni α'] = df_details['Bonferroni α'].map(lambda x: '{:5.3e}'.format(x))

    try:
        if average_multi_hypothesis:
            df_details = df_details[['Modality',
                                    'Word embedding',
                                    'Cognitive source',
                                    'Features',
                                    'Hypotheses',
                                    'Ø MSE',
                                    'SD MSE',
                                    'α',
                                    'Bonferroni α',
                                    'p',
                                    'Details',
                                    'significant']]
        else:
            df_details = df_details[['Modality',
                                    'Word embedding',
                                    'Cognitive source',
                                    'Subject',
                                    'Feature',
                                    'Ø MSE',
                                    'SD MSE',
                                    'α',
                                    'Bonferroni α',
                                    'p',
                                    'Details',
                                    'significant']]
    except KeyError:
        if average_multi_hypothesis:
            df_details = df_details[['Modality',
                                     'Word embedding',
                                     'Cognitive source',
                                     'Features',
                                     'Hypotheses',
                                     'Ø MSE',
                                     'SD MSE',
                                     'Details']]
        else:
            df_details = df_details[['Modality',
                                    'Word embedding',
                                    'Cognitive source',
                                    'Subject',
                                    'Feature',
                                    'Ø MSE',
                                    'SD MSE',
                                    'Details']]
    if not features:
        try:
            df_details = df_details.drop(['Feature'], axis='columns')
        except KeyError:
            df_details = df_details.drop(['Features'], axis='columns')

    # Detail (random)
    results = []
    for experiment, exp_file in experiment_to_path.items():
        if 'random' in experiment:
            try:
                with open(exp_file) as f:
                    result_dict = json.load(f)
                result = {'Modality': MODALITIES_SHORT_TO_FULL[result_dict['modality']],
                          'Ø MSE': result_dict['AVERAGE_MSE'],
                          'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                          'Word embedding': result_dict['wordEmbedding'],
                          'Subject': result_dict['cognitiveData'] if result_dict['multi_hypothesis'] == 'subject' else '-',
                          'Cognitive source': result_dict['cognitiveParent'],
                          'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature'],
                          'Details': result_dict['details']}
                results.append(result)
            except FileNotFoundError:
                pass

    if results:
        df_random = pd.DataFrame(results)
        numtypes = ['Ø MSE', 'SD MSE']
        group_by_keys = ['Modality', 'Word embedding', 'Cognitive source', 'Details']
        
        df_random = df_random.astype({k:float for k in numtypes})

        if average_multi_hypothesis:
            df_random_atomic = df_random.copy()[(df_random['Subject'] == '-') & (df_random['Feature'] == '-')]
            df_random_feat_avg = df_random.copy()[(df_random['Subject'] == '-') & (df_random['Feature'] != '-')]
            df_random_multi_subj_avg = df_random.copy()[df_random['Subject'] != '-']
            df_random_atomic = df_random_atomic.drop(['Subject'], axis='columns')
            df_random_feat_avg = df_random_feat_avg.drop(['Subject'], axis='columns')
            df_random_atomic = df_random_atomic.groupby(group_by_keys).mean()
            df_random_atomic['Hypotheses'] = 1
            df_random_atomic['Features'] = '-'
            df_random_feat_avg = df_random_feat_avg.groupby(group_by_keys) \
                                                    .agg({'Feature': lambda col: ', '.join(col),
                                                        **{k:'mean' for k in numtypes}}) \
                                                    .rename(columns={'Feature': 'Features'})
            df_random_feat_avg['Hypotheses'] = df_random_feat_avg['Features'].apply(lambda x: len(x.split(', ')))

            df_random_multi_subj_avg = df_random_multi_subj_avg.groupby(group_by_keys) \
                                                                .agg({'Subject': 'size',
                                                                    **{k:'mean' for k in numtypes}}) \
                                                                .rename(columns={'Subject': 'Hypotheses'})
            df_random_multi_subj_avg['Features'] = '-'
            df_random = pd.concat([df_random_atomic, df_random_feat_avg, df_random_multi_subj_avg])
            df_random['Hypotheses'] = df_random['Hypotheses'].astype(int)
            df_random.reset_index(inplace=True)
        for col in ['Ø MSE', 'SD MSE']:
            try:
                df_random[col] = df_random[col].map(lambda x: ('{:.%df}' % precision).format(x))
            except KeyError:
                pass
        
        if average_multi_hypothesis:
            df_random = df_random[['Modality',
                                'Word embedding',
                                'Cognitive source',
                                'Features',
                                'Hypotheses',
                                'Ø MSE',
                                'SD MSE',
                                'Details']]
        else:
            df_random = df_random[['Modality',
                                   'Word embedding',
                                   'Cognitive source',
                                   'Subject',
                                   'Feature',
                                   'Ø MSE',
                                   'SD MSE',
                                   'Details']]
        if not features:
            try:
                df_random = df_random.drop(['Feature'], axis='columns')
            except KeyError:
                df_random = df_random.drop(['Features'], axis='columns')
    else:
        df_random = None

    # Aggregated
    agg_modality_to_max_run_id = {}

    df_agg_dict = {}
    df_agg_for_plot = None
    df_agg_for_plot_rows = []

    if agg_reports_dict:
        for modality, mod_report_run_ids in agg_reports_dict.items():
            mod_report = mod_report_run_ids.get(run_id, None)
            if not mod_report:
                continue
            mod_report_formatted = deepcopy(mod_report)
            for col in ['ØØ MSE Baseline',  'ØØ MSE Embeddings']:
                for k, v in mod_report_formatted[col].items():
                    mod_report_formatted[col][k] = ('{:.%df}' % precision).format(v)

            agg_modality_to_max_run_id[modality] = run_id
            df_agg = pd.DataFrame(mod_report_formatted)
            df_agg_num = pd.DataFrame(mod_report)

            df_agg.reset_index(inplace=True)
            df_agg.rename(columns={'index':  'Embeddings'}, inplace=True)
            df_agg_num.reset_index(inplace=True)
            df_agg_num.rename(columns={'index': 'Embeddings'}, inplace=True)
            df_agg = df_agg[['Embeddings', 'ØØ MSE Baseline', 'ØØ MSE Embeddings', 'Significance']]
            df_agg_dict[MODALITIES_SHORT_TO_FULL[modality]] = df_agg

            for _, row in df_agg_num.iterrows():
                row_proper = {'Modality': MODALITIES_SHORT_TO_FULL[modality],
                            'Embeddings': row['Embeddings'],
                            'MSE CV folds': row['Fold errors Embeddings'],
                            'Type': 'proper',
                            'Significance': row['Significance']}

                row_random = {'Modality': MODALITIES_SHORT_TO_FULL[modality],
                            'Embeddings': row['Embeddings'],
                            'MSE CV folds': row['Fold errors Baseline'],
                            'Type': 'random',
                            'Significance': '-'}

                df_agg_for_plot_rows.extend([row_proper, row_random])

        if df_agg_for_plot_rows:
            df_agg_for_plot = pd.DataFrame(df_agg_for_plot_rows)

            df_list = [pd.DataFrame(y) for x, y in df_agg_for_plot.groupby('Modality', as_index=False)]

            sig_stats_plots = []
            for df_agg_for_plot in df_list:
                try:
                    sig_stats_plots.append((df_agg_for_plot['Modality'].values[0],
                                            sig_bar_plot(df_agg_for_plot)))
                except RuntimeError:
                    pass
        
            # Generate stats over time plots if run_id > 1
            if run_id > 1:
                try:
                    stats_over_time_plots = agg_stats_over_time_plots(agg_reports_dict, run_id)
                except:
                    stats_over_time_plots = None
            else:
                stats_over_time_plots = None
        else:
            sig_stats_plots = None
            stats_over_time_plots =None
    else:
        sig_stats_plots = None
        stats_over_time_plots = None
    html_str = template.render(float64=np.float64,
                               title='CogniVal Report (Configuration {})'.format(configuration),
                               average_multi_hypothesis=average_multi_hypothesis,
                               config_dict=config_dict_report,
                               training_history_plots=training_history_plots,
                               stats_plots=sig_stats_plots,
                               stats_over_time_plots=stats_over_time_plots,
                               df_details=df_details,
                               df_agg_dict=df_agg_dict,
                               df_random=df_random,
                               avg_error_dfs=avg_error_dfs)
    
    timestamp = datetime.datetime.now().isoformat()
    f_path_html = report_dir / 'cognival_report_{}_{}.html'.format(run_id, timestamp)
    with open(f_path_html, 'w') as f:
        f.write(html_str)

    if html:
        cprint('Saved HTML report in: {}'.format(f_path_html), 'green')

        if open_html:
            url = 'file://' + str(f_path_html)
            webbrowser.open(url)

    if pdf:
        f_path_pdf = report_dir / 'cognival_report_{}_{}.pdf'.format(run_id, timestamp)
        pdfkit.from_file(str(f_path_html),
                         str(f_path_pdf),
                         options={'quiet': '',
                                  'print-media-type': ''})
        cprint('Saved PDF report in: {}'.format(f_path_pdf), 'green')

        if open_pdf:
            try:
                Popen(['xdg-open', f_path_pdf], stdout=DEVNULL, stderr=DEVNULL)
            except FileNotFoundError:
                try:
                    Popen(['gio', f_path_pdf], stdout=DEVNULL, stderr=DEVNULL)
                except FileNotFoundError:
                    try:
                        Popen(['gvfs-open', f_path_pdf], stdout=DEVNULL, stderr=DEVNULL)
                    except FileNotFoundError:
                        cprint("Cannot automatically open generated pdf file, skipping ...", "magenta")
                        return
