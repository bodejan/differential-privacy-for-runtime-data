import os
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot
import pandas as pd
from itertools import product
import warnings
from joblib import Parallel, delayed
import json

def grid_search():
    """
    Perform a grid search to find optimal hyperparameters for TVAESynthesizer.
    """
    warnings.filterwarnings('ignore')

    # Define the parameter grid
    param_grid = {
        'enforce_min_max_values': [True, False],
        'enforce_rounding': [True],
        'epochs': [300, 1000, 1500],
        'batch_size': [200, 500, 800],
        'compress_dims': [(64, 64), (128, 128), (256, 256)],
        'decompress_dims': [(64, 64), (128, 128), (256, 256)],
        'embedding_dim': [64, 128, 256],
        'l2scale': [1e-5, 1e-4, 1e-3],
        'loss_factor': [1, 2, 5]
    }

    # Import data
    file = 'grep'
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, f"c3o_data/{file}.tsv")
    data = pd.read_csv(path, sep='\t')

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    metadata.validate()

    grid_combinations = list(product(*param_grid.values()))
    grid_combinations_dict = []

    # Create combinations with metadata
    for combination in grid_combinations:
        combination_dict = dict(zip(param_grid.keys(), combination))
        combination_dict['metadata'] = metadata
        grid_combinations_dict.append(combination_dict)

    result = []
    grid_combinations_dict_len = len(grid_combinations_dict)
    num_cores = 4  # Number of cores to use for parallel execution

    processed_results = Parallel(n_jobs=num_cores)(
        delayed(process_combination)(index, combination, data, metadata, grid_combinations_dict_len)
        for index, combination in enumerate(grid_combinations_dict)
    )

    result.extend(processed_results)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, 'result_c3o.json')

    with open(path, 'w') as file:
        json.dump(result, file)

def process_combination(index, combination, data, metadata, grid_combinations_dict_len):
    """
    Process a combination of hyperparameters for TVAESynthesizer.

    Args:
        index (int): Index of the combination.
        combination (dict): Hyperparameter combination.
        data: Real data.
        metadata: Metadata for the dataset.
        grid_combinations_dict_len (int): Total number of combinations.

    Returns:
        dict: Combination and its evaluation score.
    """
    try:
        synthesizer = TVAESynthesizer(
            metadata=metadata,
            enforce_min_max_values=combination['enforce_min_max_values'],
            epochs=combination['epochs'],
            batch_size=combination['batch_size'],
            compress_dims=combination['compress_dims'],
            decompress_dims=combination['decompress_dims'],
            embedding_dim=combination['embedding_dim'],
            l2scale=combination['l2scale'],
            loss_factor=combination['loss_factor']
        )
        synthesizer.fit(data)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_directory, f'sample/{index}')
        synthetic_data = synthesizer.sample(num_rows=1000, output_file_path=path);

        quality_report = evaluate_quality(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            verbose=False
        )
        score = quality_report.get_score()
        print(f'{index+1}/{grid_combinations_dict_len}: {score}')
        os.remove(path)
        combination.pop('metadata')
        return {'combination': combination, 'score': score}
    except Exception as e:
        print(e)
        combination.pop('metadata')
        return {'combination': combination, 'score': 0}

def top10_config_for_file(file):
    """
    Generate and evaluate top 10 configurations for a given file.

    Args:
        file (str): Name of the dataset file.

    Returns:
        list: Scores of the top 10 configurations.
    """
    # Load the JSON file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, f'c3o_TVAE_{file}.json')
    with open(path, 'r') as f:
        data = json.load(f)

    # Extract the score and configuration for each configuration
    scores = [config['score'] for config in data]
    configurations = [config['combination'] for config in data]

    # Sort the configurations by score in descending order
    sorted_scores = sorted(scores, reverse=True)
    sorted_configurations = [configurations[scores.index(score)] for score in sorted_scores]

    # Import data
    path = os.path.join(script_directory, f"c3o_data/{file}.tsv")
    data = pd.read_csv(path, sep='\t')
        
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    metadata.validate()

    scores = []
    scores.append(file)

    for combination in sorted_configurations:
        synthesizer = TVAESynthesizer(
                metadata=metadata,
                enforce_min_max_values=combination['enforce_min_max_values'],
                epochs=combination['epochs'],
                batch_size=combination['batch_size'],
                compress_dims=combination['compress_dims'],
                decompress_dims=combination['decompress_dims'],
                embedding_dim=combination['embedding_dim'],
                l2scale=combination['l2scale'],
                loss_factor=combination['loss_factor']
            )
        synthesizer.fit(data)
        synthetic_data = synthesizer.sample(num_rows=1000)
        quality_report = evaluate_quality(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            verbose=False
        )
        score = quality_report.get_score()
        scores.append(score)

    return scores

def top10_config_for_all_files():
    """
    Generate and print the top 10 configurations for all files.
    """
    grep = top10_config_for_file('grep')
    print('grep')
    kmeans = top10_config_for_file('kmeans')
    print('kmeans')
    pagerank = top10_config_for_file('pagerank')
    print('pagerank')
    sgd = top10_config_for_file('sgd')
    print('sgd')
    sort = top10_config_for_file('sort')
    print('sort')

    data = {}

    # Create a DataFrame with top 10 scores for each file
    for array in [grep, kmeans, pagerank, sgd, sort]:
        column_name = array[0]
        column_data = array[1:]
        data[column_name] = column_data

    df = pd.DataFrame(data)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_directory, 'c3o_TVAE_top10.csv')
    df.to_csv(path)

top10_config_for_all_files()
