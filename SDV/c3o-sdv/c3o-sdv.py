import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
import csv

def gen_synthetic_data(file, sample_size):
    reports = []
    # import data
    data = pd.read_csv(f'synthetic-data/c3o-sdv/c3o-data/{file}.tsv', sep='\t')
    # print(data.head())
    
    # detect metadata & validate metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    dict = metadata.to_dict()
    print(dict)
    metadata.validate()

    # train models 
    # DayZSynthesizer is only available for enterprise user; model generates data from scratch using only metadata
    # synthesizer = [SingleTablePreset(metadata, name='FAST_ML'), CopulaGANSynthesizer(metadata), CTGANSynthesizer(metadata), DayZSynthesizer(metadata), GaussianCopulaSynthesizer(metadata), TVAESynthesizer(metadata)]
    synthesizers = [SingleTablePreset(metadata, name='FAST_ML'), CopulaGANSynthesizer(metadata), CTGANSynthesizer(metadata), GaussianCopulaSynthesizer(metadata), TVAESynthesizer(metadata)]
    for s in synthesizers: 
        s.fit(data)

        # create synthetic sample 
        synthetic_data = s.sample(num_rows = sample_size)
        synthetic_data.to_csv(f'synthetic-data/c3o-sdv/c3o-data-synthetic/{file}/{file}_{s.__class__.__name__}.tsv', sep='\t')

        # TODO advanced sampling options

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
    
        reports.append({'quality_report': quality_report, 'diagnostic_report': diagnostic_report, 'class': s.__class__.__name__, 'file': file})

    return reports
    

    # visualize
    #fig = get_column_plot(
    #    real_data=data,
    #    synthetic_data=synthetic_data,
    #    column_name='amenities_fee',
    #    metadata=metadata
    #)
    # fig.show()

files = ['grep', 'kmeans', 'pagerank', 'sgd', 'sort']
file_reports = []
for f in files:
    file_report = gen_synthetic_data(f, 200)
    file_reports.append(file_report)

for file_report in file_reports:
    for report in file_report: 
        row = []
        row.append(report['file'])
        row.append(report['class'])
        row.append(report['quality_report'].get_score())
        with open('synthetic-data/c3o-sdv/results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['file', 'model', 'score'])
            writer.writerow(row)
        #print(report['file'], '|', report['class'], '|', report['quality_report'].get_score())