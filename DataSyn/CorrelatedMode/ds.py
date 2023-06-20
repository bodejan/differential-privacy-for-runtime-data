from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd

class DS():
        
    def __init__(self, mode: str ='correlated_attribute_mode', epsilon: int = 0, num_tuples:int = 1000, input_data:str = "'../../datasets/c3o-experiments/sort.csv'"):

        description_file = f'description.json'
        synthetic_data = f'sythetic_sortdata.csv'

        # An attribute is categorical if its domain size is less than this threshold.
        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
        threshold_value = 10
        categorical_attributes = {'machine_type': True}

        # specify which attributes are candidate keys of input dataset.
        candidate_keys = {'machine_type': False}

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = 2

        # Number of tuples generated in synthetic dataset.
        num_tuples_to_generate = 1000 # Here 32561 is the same as input dataset, but it can be set to another number.


        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                            epsilon=epsilon, 
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys)
        describer.save_dataset_description_to_file(description_file)


        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        generator.save_synthetic_data(synthetic_data)
        
        # Read both datasets using Pandas.
        input_df = pd.read_csv(input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)
        # Read attribute description from the dataset description file.
        attribute_description = read_json_file(description_file)['attribute_description']
        
        inspector = ModelInspector(input_df, synthetic_df, attribute_description)
        
        inspector.mutual_information_heatmap()