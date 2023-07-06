
from DataSynthesizer.lib.utils import pairwise_attributes_mutual_information, normalize_given_distribution
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp


class IDS():
    def request(self,input_data:str,uuid:str, epsilon: int = 0, num_tuples: int = 1000,  dataset: str="sort"):
        print(input_data)
        description_file = f'dashboard/temp/{uuid}.json'
        synthetic_data = f'dashboard/temp/{uuid}.csv'

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
        describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data,
                                                         attribute_to_is_categorical=categorical_attributes,
                                                         attribute_to_is_candidate_key=candidate_keys)
        
        describer.save_dataset_description_to_file(description_file)
        print("save")

        generator = DataGenerator()
        generator.generate_dataset_in_independent_mode(num_tuples, description_file)
        generator.save_synthetic_data(synthetic_data)
        
        print("generator")

        # Read both datasets using Pandas.
        input_df = pd.read_csv(input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)
        # Read attribute description from the dataset description file.
        #attribute_description = read_json_file(description_file)['attribute_description']

        # Calculate mutual information
        private_mi = pairwise_attributes_mutual_information(input_df)
        synthetic_mi = pairwise_attributes_mutual_information(synthetic_df)

        # Create subplots
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Private, max=1", "Synthetic, max=1"))

        # Add heatmaps
        fig.add_trace(go.Heatmap(z=private_mi, colorscale="Blues"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=synthetic_mi, colorscale="Blues"), row=1, col=2)

        # Update layout
        fig.update_layout(
            title="Pairwise Mutual Information Comparison (Private vs Synthetic)",
            width=900,
            height=400,
            showlegend=False
        )
        return fig