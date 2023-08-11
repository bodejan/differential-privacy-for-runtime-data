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
    """
    Generate synthetic data, evaluate quality, and perform diagnostics for a given dataset file.

    Args:
        file (str): Name of the dataset file.
        sample_size (int): Number of rows in the synthetic dataset.

    Returns:
        list: List of dictionaries containing quality and diagnostic reports for each synthesizer.
    """
    reports = []

    # Import data
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, f'c3o_data/{file}.tsv')
    data = pd.read_csv(path, sep='\t')

    # Detect metadata & validate metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    metadata.validate()

    # Train models
    synthesizers = [
        SingleTablePreset(metadata, name='FAST_ML'),
        CopulaGANSynthesizer(metadata),
        CTGANSynthesizer(metadata),
        GaussianCopulaSynthesizer(metadata),
        TVAESynthesizer(metadata)
    ]
    for s in synthesizers: 
        s.fit(data)

        # Create synthetic sample
        synthetic_data = s.sample(num_rows=sample_size)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_directory, f'c3o_data_synthetic/{file}/{file}_{s.__class__.__name__}.tsv')
        synthetic_data.to_csv(path, sep='\t')

        # Evaluate
        quality_report = evaluate_quality(data, synthetic_data, metadata)

        # Diagnostics
        diagnostic_report = run_diagnostic(real_data=data, synthetic_data=synthetic_data, metadata=metadata)

        reports.append({'quality_report': quality_report, 'diagnostic_report': diagnostic_report, 'class': s.__class__.__name__})

    return reports
    

def write_results(file_reports): 
    """
    Write evaluation results to a CSV file.

    Args:
        file_reports (list): List of dictionaries containing evaluation reports.
    """
    for file_report in file_reports:
        for report in file_report: 
            row = [report['file'], report['class'], report['quality_report'].get_score()]
            script_directory = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_directory, 'results.csv')
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['file', 'model', 'score'])
                writer.writerow(row)
            print(report['file'], '|', report['class'], '|', report['quality_report'].get_score())

def column_plot(data, synthetic_data, metadata):
    """
    Generate a plot for a specific column in the dataset.

    Args:
        data: Real data.
        synthetic_data: Synthetic data.
        metadata: Metadata for the dataset.
    """
    columns = metadata.columns.keys()
    first_key = next(iter(columns))
    # Visualize
    fig = get_column_plot(real_data=data, synthetic_data=synthetic_data, column_name=first_key, metadata=metadata)
    fig.show()

def plot_all_columns(data, synthetic_data, metadata):
    """
    Generate plots for all columns in the dataset.

    Args:
        data: Real data.
        synthetic_data: Synthetic data.
        metadata: Metadata for the dataset.
    """
    columns = metadata.columns.keys()
    all_figs = make_subplots(rows=len(columns), cols=1)
    for c in columns:
        fig = get_column_plot(real_data=data, synthetic_data=synthetic_data, column_name=c, metadata=metadata)
        all_figs.add_trace(fig.data)
    fig.show()

files = ['grep', 'kmeans', 'pagerank', 'sgd', 'sort']
file_reports = []

# Generate synthetic data, evaluate, and store reports for each file
for f in files:
    file_report = gen_synthetic_data(f, 200)
    file_reports.append(file_report)

# Write evaluation results to CSV
write_results(file_reports)
