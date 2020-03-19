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
from jinja2 import Environment, PackageLoader, select_autoescape
import pdfkit
from operator import truediv as div
from subprocess import Popen, DEVNULL

import matplotlib.pyplot as plt
import seaborn as sns
import random

from termcolor import cprint
from .utils import _open_config

# Set seaborn figure size and font scale
xkcd_colors = [v for k, v in sns.xkcd_rgb.items() if not any([prop in k for prop in ['faded', 'light', 'pale', 'dark']])]
random.seed(48)
random.shuffle(xkcd_colors)

# Set style
sns.set(style="whitegrid", color_codes=True)
sns.set_context('paper')

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


def sig_bar_plot(df, max_y=1.0):
    '''
    Generates a bar plot with average MSE and significance stats per embedding type.
    '''
    sig_labels = [row['Significance'] for _, row in df.iterrows() if row['Significance'] != '-']
    # Make the barplot
    with sns.plotting_context("paper", font_scale=1.1):
        bar = sns.catplot(x="Embedding", y="Ø MSE", hue="Type", kind="bar", data=df, palette=["C1", "#dddddd"], legend=False)
        bar.ax.set(ylim=(0, max_y + 0.1))

        # Loop over the bars
        patches = list(bar.ax.patches)
        half_patch_num = len(patches)//2

        # proper embeddings (get xkcd colors)
        for idx, thisbar in enumerate(patches[:half_patch_num]):
            thisbar.set_width(0.95 * thisbar.get_width())
            thisbar.set_color(xkcd_colors[idx])
            x = thisbar.get_x()
            y = thisbar.get_height() + 0.01
            bar.ax.annotate('({})\n{:.2f}'.format(sig_labels[idx], thisbar.get_height()), (x, y))   

        # random embeddings (grey)
        for idx, thisbar in enumerate(patches[half_patch_num:]):
            thisbar.set_width(0.95 * thisbar.get_width())
            thisbar.set_hatch('-')
            x = thisbar.get_x()
            y = thisbar.get_height() + 0.01
            bar.ax.annotate('{:.2f}'.format(thisbar.get_height()), (x, y)) 
        
        # Adjust the margins
        plt.subplots_adjust(bottom= 0.2, top = 0.8)
        plt.xticks(rotation=45)

        with BytesIO() as figfile:
            bar.fig.set_size_inches(18, 8)
            plt.savefig(figfile, format='png', dpi=300, bbox_inches="tight")
            figfile.seek(0)  # rewind to beginning of file
            statsfig_b64 = base64.b64encode(figfile.getvalue()).decode('utf8')
    return statsfig_b64

def agg_stats_over_time_plots(agg_reports_dict, run_id):
    '''
    Generates line plots with aggregate stats over time (run_ids)
    Ø MSE Baseline, Ø MSE Proper, Significance
    '''
    df_list = []
    modality_to_plots = {}
    for modality, run_ids in agg_reports_dict.items():
        for agg_run_id, agg_params in run_ids.items():
            df = pd.DataFrame.from_dict(agg_params)
            df.reset_index(inplace=True)
            df.rename(columns={'index':'Embeddings'}, inplace=True)
            df['run_id'] = [agg_run_id]*len(df)
            df['Significance'] = df['Significance'].apply(lambda x: div(*map(int, x.split('/'))))
            df_list.append(df)
        df = pd.concat(df_list)
        df = df[df['run_id'] <= run_id]

        plots_b64 = []
        for measure in ["Ø MSE Baseline", "Ø MSE Proper", "Significance"]:
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
                    run_id,
                    resources_path,
                    precision,
                    average_multi_hypothesis,
                    include_training_history_plots,
                    include_features,
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

    experiments_dir = out_dir / 'experiments'
    sig_test_res_dir = out_dir / 'sig_test_results'
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
            if include_training_history_plots:
                with open(experiments_dir / value['proper'] / '{}.png'.format(value['embedding']), 'rb') as f:
                    figdata_b64 = base64.b64encode(f.read()).decode('utf8')
                    training_history_plots[key] = figdata_b64
            else:
                training_history_plots = None
        except FileNotFoundError:
            continue

    # Collecting significance test results and aggregation results
    for path, _, reports in os.walk(report_dir):
        if reports:
            for report in reports:
                if not any(report.endswith(suffix) for suffix in ('html, pdf')):
                    modality, ver = path.split('/')[-2:]
                    with open(Path(path) / report) as f_sig:
                        report_dict = json.loads(f_sig.read())
                    if report == 'Wilcoxon.json':
                        sig_test_reports_dict[modality][int(ver)] = report_dict
                    else:
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
                result = {'Experiment': experiment,
                        'Modality': MODALITIES_SHORT_TO_FULL[modality],
                        'Ø MSE': result_dict['AVERAGE_MSE'],
                        'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                        'Word embedding': result_dict['wordEmbedding'],
                        'Subject': result_dict['cognitiveData'] if result_dict['cognitiveData'] != result_dict['cognitiveParent'] else '-',
                        'Cognitive source': result_dict['cognitiveParent'],
                        'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature'],
                        **sig_test_result}
                results.append(result)
    # If no significance tests have been performed
    if not results:
        for experiment, result_json_path in experiment_to_path.items():
            with open(result_json_path) as f:
                result_dict = json.load(f)
                result = {'Experiment': experiment,
                        'Modality': MODALITIES_SHORT_TO_FULL[result_dict['modality']],
                        'Ø MSE': result_dict['AVERAGE_MSE'],
                        'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                        'Word embedding': result_dict['wordEmbedding'],
                        'Subject': result_dict['cognitiveData'] if result_dict['cognitiveData'] != result_dict['cognitiveParent'] else '-',
                        'Cognitive source': result_dict['cognitiveParent'],
                        'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature']}
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
        group_by_keys = ['Modality', 'Word embedding', 'Cognitive source']
        df_details_atomic = df_details.copy()[(df_details['Subject'] == '-') & (df_details['Feature'] == '-')]
        df_details_feat_avg = df_details.copy()[(df_details['Subject'] == '-') & (df_details['Feature'] != '-')]
        df_details_multi_subj_avg = df_details.copy()[df_details['Subject'] != '-']
        df_details_atomic = df_details_atomic.drop(['Experiment', 'Subject'], axis='columns')
        df_details_feat_avg = df_details_feat_avg.drop(['Experiment', 'Subject'], axis='columns')
        df_details_multi_subj_avg = df_details_multi_subj_avg.drop(['Experiment'], axis='columns')
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
                                    'significant']]
        else:
            df_details = df_details[['Modality',
                                    'Experiment',
                                    'Word embedding',
                                    'Cognitive source',
                                    'Subject',
                                    'Feature',
                                    'Ø MSE',
                                    'SD MSE',
                                    'α',
                                    'Bonferroni α',
                                    'p',
                                    'significant']]
    except KeyError:
        if average_multi_hypothesis:
            df_details = df_details[['Modality',
                                     'Word embedding',
                                     'Cognitive source',
                                     'Features',
                                     'Hypotheses',
                                     'Ø MSE',
                                     'SD MSE']]
        else:
            df_details = df_details[['Modality',
                                    'Experiment',
                                    'Word embedding',
                                    'Cognitive source',
                                    'Subject',
                                    'Feature',
                                    'Ø MSE',
                                    'SD MSE']]
    if not include_features:
        try:
            df_details = df_details.drop(['Feature'], axis='columns')
        except KeyError:
            df_details = df_details.drop(['Features'], axis='columns')

    # Detail (random)
    results = []
    for experiment, exp_file in experiment_to_path.items():
        if 'random' in experiment:
            with open(exp_file) as f:
                result_dict = json.load(f)
            result = {'Experiment': random_to_proper[experiment],
                      'Modality': MODALITIES_SHORT_TO_FULL[result_dict['modality']],
                      'Ø MSE': result_dict['AVERAGE_MSE'],
                      'SD MSE': np.std([x['MSE_PREDICTION'] for x in result_dict['folds']]),
                      'Word embedding': result_dict['wordEmbedding'],
                      'Subject': result_dict['cognitiveData'] if result_dict['cognitiveData'] != result_dict['cognitiveParent'] else '-',
                      'Cognitive source': result_dict['cognitiveParent'],
                      'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature']}
            results.append(result)

    if results:
        df_random = pd.DataFrame(results)
        numtypes = ['Ø MSE', 'SD MSE']
        group_by_keys = ['Modality', 'Word embedding', 'Cognitive source']
        
        df_random = df_random.astype({k:float for k in numtypes})

        if average_multi_hypothesis:
            df_random_atomic = df_random.copy()[(df_random['Subject'] == '-') & (df_random['Feature'] == '-')]
            df_random_feat_avg = df_random.copy()[(df_random['Subject'] == '-') & (df_random['Feature'] != '-')]
            df_random_multi_subj_avg = df_random.copy()[df_random['Subject'] != '-']
            df_random_atomic = df_random_atomic.drop(['Experiment', 'Subject'], axis='columns')
            df_random_feat_avg = df_random_feat_avg.drop(['Experiment', 'Subject'], axis='columns')
            df_random_multi_subj_avg = df_random_multi_subj_avg.drop(['Experiment'], axis='columns')
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
                                'SD MSE']]
        else:
            df_random = df_random[['Modality',
                                   'Experiment',
                                   'Word embedding',
                                   'Cognitive source',
                                   'Subject',
                                   'Feature',
                                   'Ø MSE',
                                   'SD MSE']]
        if not include_features:
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
            for col in ['Ø MSE Baseline', 'Ø MSE Proper']:
                for k, v in mod_report_formatted[col].items():
                    mod_report_formatted[col][k] = ('{:.%df}' % precision).format(v)
            agg_modality_to_max_run_id[modality] = run_id
            df_agg = pd.DataFrame(mod_report_formatted)
            df_agg_num = pd.DataFrame(mod_report)

            df_agg.reset_index(inplace=True)
            df_agg.rename(columns={'index': 'Word embedding'}, inplace=True)
            df_agg_num.reset_index(inplace=True)
            df_agg_num.rename(columns={'index': 'Word embedding'}, inplace=True)
            df_agg = df_agg[['Word embedding', 'Ø MSE Baseline', 'Ø MSE Proper', 'Significance']]
            df_agg_dict[MODALITIES_SHORT_TO_FULL[modality]] = df_agg

            for _, row in df_agg_num.iterrows():
                row_proper = {'Modality': MODALITIES_SHORT_TO_FULL[modality],
                            'Embedding': row['Word embedding'],
                            'Ø MSE': row['Ø MSE Proper'],
                            'Type': 'proper',
                            'Significance': row['Significance']}

                row_random = {'Modality': MODALITIES_SHORT_TO_FULL[modality],
                            'Embedding': row['Word embedding'],
                            'Ø MSE': row['Ø MSE Baseline'],
                            'Type': 'random',
                            'Significance': '-'}

                df_agg_for_plot_rows.extend([row_proper, row_random])

        if df_agg_for_plot_rows:
            df_agg_for_plot = pd.DataFrame(df_agg_for_plot_rows)
            max_y = df_agg_for_plot['Ø MSE'].max()

            df_list = [pd.DataFrame(y) for x, y in df_agg_for_plot.groupby('Modality', as_index=False)]

            sig_stats_plots = []
            for df_agg_for_plot in df_list:
                sig_stats_plots.append((df_agg_for_plot['Modality'].values[0], sig_bar_plot(df_agg_for_plot, max_y=max_y)))
        
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
                               df_random=df_random)
    
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