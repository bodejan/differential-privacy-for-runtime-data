import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import pairwise_attributes_mutual_information

from synthesizer import Synthesizer


class CorrelatedDataSynthesizer(Synthesizer):
    """
    Class containing all functions to run the synthetic data synthesizer in correlated mode
    """
    def request(self, epsilon: int = 0, num_tuples: int = 1000, dataset: str = "sort", **_):
        """
        Train bayesian network based on given parameters
        :param epsilon: Value for epsilon: 0 == much privacy, <15 == no privacy
        :param num_tuples: Number of Samples to generate
        :param dataset: Dataset to synthesize
        :param _: ignored parameters that are not available in correlated mode
        :return: Figure showing quality of synthetic model
        """

        # define location to store synthetic data and data description file
        description_file = f'temp/{self._session_id}.json'
        synthetic_data = f'temp/{self._session_id}.csv'

        threshold_value = 10 # threshold to define domain size of categorical attributes
        categorical_attributes = {'machine_type': True}

        candidate_keys = {'machine_type': False} # define possible primary keys in data that should be ignored

        degree_of_bayesian_network = 2 # degree of bayesian network, i.e. maximum number of parents for each node

        # create and save data describer
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=self._input_data,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network,
                                                                attribute_to_is_categorical=categorical_attributes,
                                                                attribute_to_is_candidate_key=candidate_keys)
        describer.save_dataset_description_to_file(description_file)

        # create data generator and create synthetic data
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples, description_file)
        generator.save_synthetic_data(synthetic_data)

        # calculate correlation matrix from synthetic & normal data
        input_df = pd.read_csv(self._input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)

        # calculate mutual information with function given in DataSynthesizer
        private_mi = pairwise_attributes_mutual_information(input_df)
        synthetic_mi = pairwise_attributes_mutual_information(synthetic_df)

        # create subplots
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Private, max=1", "Synthetic, max=1"))

        # add heatmaps
        fig.add_trace(go.Heatmap(z=private_mi, colorscale="Blues"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=synthetic_mi, colorscale="Blues"), row=1, col=2)

        # update layout
        fig.update_layout(
            title="Pairwise Mutual Information Comparison (Private vs Synthetic)",
            width=900,
            height=400,
            showlegend=False
        )
        return fig
