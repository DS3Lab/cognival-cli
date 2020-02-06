import datetime
import collections
import json
import tempfile
import webbrowser
import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
import pdfkit
from collections import namedtuple
from operator import truediv as div
from subprocess import Popen, DEVNULL

from numpy import exp, cos, linspace
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(48)

from termcolor import cprint

from .utils import (tupleit,
                   _open_config,
                   _open_cog_config,
                   _check_cog_installed,
                   _check_emb_installed,
                   _save_cog_config,
                   _save_config,
                   DisplayablePath,
                   download_file,
                   AbortException,
                   chunked_list_concat_str,
                   field_concat,
                   chunks,
                   page_list)

# Set seaborn figure size and font scale
xkcd_colors = [v for k, v in sns.xkcd_rgb.items() if not any([prop in k for prop in ['faded', 'light', 'pale', 'dark']])]
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

    template = env.get_template(template_file)
    return template

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
            bar.ax.annotate('{:.2f} ({})'.format(thisbar.get_height(), sig_labels[idx]), (x, y))
            

        # random embeddings (grey)
        for idx, thisbar in enumerate(patches[half_patch_num:]):
            thisbar.set_width(0.95 * thisbar.get_width())
            thisbar.set_hatch('-')
            x = thisbar.get_x()
            y = thisbar.get_height() + 0.01
            bar.ax.annotate('{:.2f}'.format(thisbar.get_height()), (x, y)) 
        
        # Adjust the margins
        plt.subplots_adjust(bottom= 0.2, top = 0.8)

        with BytesIO() as figfile:
            bar.fig.set_size_inches(8, 4)
            plt.savefig(figfile, format='png', dpi=300, bbox_inches="tight")
            figfile.seek(0)  # rewind to beginning of file
            statsfig_b64 = base64.b64encode(figfile.getvalue()).decode('utf8')
    return statsfig_b64

def agg_stats_over_time_plots(agg_reports_dict):
    '''
    Generates line plots with aggregate stats over time (versions)
    Ø MSE Baseline, Ø MSE Proper, Significance
    '''
    df_list = []
    modality_to_plots = {}
    for modality, versions in agg_reports_dict.items():
        for version, agg_params in versions.items():
            for idx in range(3):
                df = pd.DataFrame.from_dict(agg_params)
                df.reset_index(inplace=True)
                df.rename(columns={'index':'Embeddings'}, inplace=True)
                df['version'] = [version]*len(df)
                df['Significance'] = df['Significance'].apply(lambda x: div(*map(int, x.split('/'))))
                df['Embeddings'] = df['Embeddings'].apply(lambda x: '{}_{}'.format(x, idx))
                df['Ø MSE Proper'] = df['Ø MSE Proper'].apply(lambda x: x * idx)
                df_list.append(df)
        df = pd.concat(df_list)

        plots_b64 = []
        for measure in ["Ø MSE Baseline", "Ø MSE Proper", "Significance"]:
            plt.clf()
            plt.cla()
            plt.figure()
            df_sub = df[['Embeddings', measure, 'version']]
            df_sub_list = []
            for emb in df_sub['Embeddings'].unique():
                df_subsub = df_sub[df_sub['Embeddings'] == emb].copy()
                df_subsub.rename(columns={measure:emb}, inplace=True)
                df_ver = df_subsub['version']
                df_subsub = df_subsub[[emb]]
                df_sub_list.append(df_subsub)
            df_sub = pd.concat(df_sub_list + [df_ver], axis = 1)
            plot = sns.lineplot(x='version', y='value', hue='variable', data=pd.melt(df_sub, ['version']))
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
                    version,
                    resources_path,
                    html=True,
                    pdf=False,
                    open_html=False,
                    open_pdf=False):
    '''
    Generates report from significance test results and aggregated statistics for given configuration
    and configuration version.
    '''
    cprint('Generating CogniVal report ...', 'green')
    template = load_jinja_template()

    config_dict = _open_config(configuration, resources_path, quiet=True)

    # Get mapping of previous version (current not yet executed)
    if not version:
        version = config_dict['version'] - 1
    out_dir = Path(config_dict['PATH']) / config_dict['outputDir']

    with open(out_dir / 'mapping_{}.json'.format(version)) as f:
        mapping_dict = json.load(f)

    report_dir = out_dir / 'reports'

    experiment_to_path = {}
    training_history_plots = {}
    random_to_proper = {}

    experiments_dir = out_dir / 'experiments'
    sig_test_res_dir = out_dir / 'sig_test_results' # / modality / str(version)
    report_dir = out_dir / 'reports' # / modality / str(version)
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
            with open(experiments_dir / value['proper'] / '{}.png'.format(value['embedding']), 'rb') as f:
                figdata_b64 = base64.b64encode(f.read()).decode('utf8')
                training_history_plots[key] = figdata_b64
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
    for modality, mod_report_versions in sig_test_reports_dict.items():
        mod_report = mod_report_versions[max(mod_report_versions)]
        bonferroni_alpha = mod_report['bonferroni_alpha']
        for experiment, sig_test_result in mod_report['hypotheses'].items():
            with open(experiment_to_path[experiment]) as f:
                result_dict = json.load(f)
            result = {'Experiment': experiment,
                      'Modality': MODALITIES_SHORT_TO_FULL[modality],
                      'Ø MSE': '{:.5f}'.format(result_dict['AVERAGE_MSE']),
                      'SD MSE': '{:.5f}'.format(np.std([x['MSE_PREDICTION'] for x in result_dict['folds']])),
                      'Word embedding': result_dict['wordEmbedding'],
                      'Cognitive data': result_dict['cognitiveData'],
                      'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature'],
                      'bonferroni_alpha':bonferroni_alpha,
                      **sig_test_result}
            result['p_value'] = '{:1.3e}'.format(result['p_value'])

            results.append(result)

    df_details = pd.DataFrame(results)
    df_details = df_details[['Modality',
                             'Experiment',
                             'Word embedding',
                             'Cognitive data',
                             'Feature',
                             'Ø MSE',
                             'SD MSE',
                             'alpha',
                             'bonferroni_alpha',
                             'p_value',
                             'significant']]

    df_details.rename(columns={'alpha': 'α',
                               'bonferroni_alpha': 'Bonferroni α',
                               'p_value': 'p'}, inplace=True)

    # Detail (random)
    results = []
    for experiment, exp_file in experiment_to_path.items():
        if 'random' in experiment:
            with open(exp_file) as f:
                result_dict = json.load(f)
            result = {'Experiment': experiment,
                      'Modality': MODALITIES_SHORT_TO_FULL[result_dict['modality']],
                      'Ø MSE': '{:.5f}'.format(result_dict['AVERAGE_MSE']),
                      'SD MSE': '{:.5f}'.format(np.std([x['MSE_PREDICTION'] for x in result_dict['folds']])),
                      'Word embedding': result_dict['wordEmbedding'],
                      'Corresponding proper': random_to_proper[experiment],
                      'Cognitive data': result_dict['cognitiveData'],
                      'Feature': '-' if result_dict['feature'] == 'ALL_DIM' else result_dict['feature']}
            results.append(result)

    df_random = pd.DataFrame(results)
    df_random = df_random[['Modality',
                           'Experiment',
                           'Corresponding proper',
                           'Word embedding',
                           'Cognitive data',
                           'Feature',
                           'Ø MSE',
                           'SD MSE']]
    # Aggregated
    agg_modality_to_max_version = {}

    df_agg_dict = {}
    df_agg_for_plot_rows = []

    for modality, mod_report_versions in agg_reports_dict.items():
        max_version = max(mod_report_versions)
        agg_modality_to_max_version[modality] = max_version
        mod_report = mod_report_versions[max_version]
        df_agg = pd.DataFrame(mod_report)

        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'index': 'Word embedding'}, inplace=True)
        df_agg = df_agg[['Word embedding', 'Ø MSE Baseline', 'Ø MSE Proper', 'Significance']]
        df_agg_dict[MODALITIES_SHORT_TO_FULL[modality]] = df_agg

        for _, row in df_agg.iterrows():
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

    df_agg_for_plot = pd.DataFrame(df_agg_for_plot_rows)
    max_y = df_agg_for_plot['Ø MSE'].max()

    df_list = [pd.DataFrame(y) for x, y in df_agg_for_plot.groupby('Modality', as_index=False)]

    sig_stats_plots = []
    for df_agg_for_plot in df_list:
        sig_stats_plots.append((df_agg_for_plot['Modality'].values[0], sig_bar_plot(df_agg_for_plot, max_y=max_y)))

    # Generate stats over time plots if more that one version
    if version > 1:
        stats_over_time_plots = agg_stats_over_time_plots(agg_reports_dict)
    else:
        stats_over_time_plots = None

    html_str = template.render(float64=np.float64,
                               title='CogniVal Report',
                               training_history_plots=training_history_plots,
                               stats_plots=sig_stats_plots,
                               stats_over_time_plots=stats_over_time_plots,
                               df_details=df_details,
                               df_agg_dict=df_agg_dict,
                               df_random=df_random)
    
    timestamp = datetime.datetime.now().isoformat()
    f_path_html = report_dir / 'cognival_report_{}_{}.html'.format(version, timestamp)
    with open(f_path_html, 'w') as f:
        f.write(html_str)
    
    if html:
        cprint('Saved HTML report in: {}'.format(f_path_html), 'green')

        if open_html:
            url = 'file://' + str(f_path_html)
            webbrowser.open(url)

    if pdf:
        f_path_pdf = report_dir / 'cognival_report_{}_{}.pdf'.format(version, timestamp)
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