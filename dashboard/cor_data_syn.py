
from DataSynthesizer.lib.utils import pairwise_attributes_mutual_information, normalize_given_distribution
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CDS():
    def __init__(self,input_data:str,uuid:str, epsilon: int = 0, num_tuples: int = 1000,  dataset: str="sort"):
        description_file = f'temp/{uuid}.json'
        synthetic_data = f'temp/{uuid}.csv'

        # An attribute is categorical if its domain size is less than this threshold.
        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
        threshold_value = 10
        categorical_attributes = {'machine_type': True}

        # specify which attributes are candidate keys of input dataset.
        candidate_keys = {'machine_type': False}

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = 2

        # Number of tuples generated in synthetic dataset.
        print("First", epsilon, num_tuples, input_data)
       
        print("Describer")
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys)
        describer.save_dataset_description_to_file(description_file)
        print("save")

        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples, description_file)
        generator.save_synthetic_data(synthetic_data)
        
        print("generator")

        # Read both datasets using Pandas.
        input_df = pd.read_csv(input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)
        # Read attribute description from the dataset description file.
        #attribute_description = read_json_file(description_file)['attribute_description']


        private_mi = pairwise_attributes_mutual_information(input_df)
        synthetic_mi = pairwise_attributes_mutual_information(synthetic_df)

        fig = plt.figure(figsize=(15, 6), dpi=120)
        fig.suptitle('Pairwise Mutual Information Comparison (Private vs Synthetic)', fontsize=20)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sns.heatmap(private_mi, ax=ax1, cmap="Blues")
        sns.heatmap(synthetic_mi, ax=ax2, cmap="Blues")
        ax1.set_title('Private, max=1', fontsize=15)
        ax2.set_title('Synthetic, max=1', fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.subplots_adjust(top=0.83)
        fig.savefig(f'assets/temp_{dataset}.png')


