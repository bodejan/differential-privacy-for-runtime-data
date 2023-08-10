import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gen_synthetic_data(file, sample_size):
    reports = []
    # import data
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, f'c3o_data/{file}.tsv')
    data = pd.read_csv(path, sep='\t')
    # print(data.head())
    
    # detect metadata & validate metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    dict = metadata.to_dict()
    #print(dict)
    metadata.validate()

    # train models 
    # DayZSynthesizer is only available for enterprise user; model generates data from scratch using only metadata
    synthesizers = [SingleTablePreset(metadata, name='FAST_ML'), CopulaGANSynthesizer(metadata), CTGANSynthesizer(metadata), GaussianCopulaSynthesizer(metadata), TVAESynthesizer(metadata)]
    for s in synthesizers: 
        s.fit(data)

        # create synthetic sample 
        synthetic_data = s.sample(num_rows = sample_size)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_directory, f'c3o_data_synthetic/{file}/{file}_{s.__class__.__name__}.tsv')
        synthetic_data.to_csv(path, sep='\t')

        # evaluate
        quality_report = evaluate_quality(
            data,
            synthetic_data,
            metadata)

        # diagnostics
        diagnostic_report = run_diagnostic(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata)

        plot_all_columns(data, synthetic_data, metadata)

        reports.append({'quality_report': quality_report, 'diagnostic_report': diagnostic_report, 'class': s.__class__.__name__, 'file': file, 'fig': fig})

    return reports
    

def write_results(file_reports): 
    for file_report in file_reports:
        for report in file_report: 
            row = []
            row.append(report['file'])
            row.append(report['class'])
            row.append(report['quality_report'].get_score())
            script_directory = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_directory, 'results.csv')
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['file', 'model', 'score'])
                writer.writerow(row)
            print(report['file'], '|', report['class'], '|', report['quality_report'].get_score())


def column_plot(data, synthetic_data, metadata):
    columns = metadata.columns.keys()
    first_key = next(iter(columns))
    # visualize
    fig = get_column_plot(
        real_data=data,
        synthetic_data=synthetic_data,
        column_name=first_key,
        metadata=metadata
    )
    fig.show()

def plot_all_columns(data, synthetic_data, metadata):
    columns = metadata.columns.keys()
    all_figs = make_subplots(rows=len(columns), cols=1)
    for c in columns:
        fig = get_column_plot(
            real_data=data,
            synthetic_data=synthetic_data,
            column_name=c,
            metadata=metadata
        )
        all_figs.add_trace(fig.data)
    fig.show()


files = ['grep', 'kmeans', 'pagerank', 'sgd', 'sort']
file_reports = []
for f in files:
    file_report = gen_synthetic_data(f, 200)
    file_reports.append(file_report)

write_results(file_reports)
