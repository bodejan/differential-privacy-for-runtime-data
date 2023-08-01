from DataSynthesizer.lib.utils import pairwise_attributes_mutual_information, normalize_given_distribution
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from synthesizer import Synthesizer


class IndependentDataSynthesizer(Synthesizer):
    """
    Class containing all functions to run the synthetic data synthesizer in independent mode
    """
    def request(self, epsilon: int = 0, num_samples: int = 1000, dataset: str = "sort", **kwargs):
        """
        Create synthetic data with independent DataSynthesizer based on given parameters
        :param epsilon: Value for epsilon: 0 == much privacy, <15 == no privacy
        :param num_samples: Number of samples to generate
        :param dataset: Dataset to create synthetic data from
        :param kwargs:
        :return: Figure showing quality of synthetic model
        """

        # define location to store synthetic data and data description file
        description_file = f'temp/{self._session_id}.json'
        synthetic_data = f'temp/{self._session_id}.csv'

        # threshold to define domain size of categorical attributes
        threshold_value = 10
        categorical_attributes = {'machine_type': True}

        # define possible primary keys in data that should be ignored
        candidate_keys = {'machine_type': False}

        # create and save data describer
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_independent_attribute_mode(dataset_file=self._input_data,
                                                                 attribute_to_is_categorical=categorical_attributes,
                                                                 attribute_to_is_candidate_key=candidate_keys)

        describer.save_dataset_description_to_file(description_file)
        print("save")

        # create data generator and create synthetic data
        generator = DataGenerator()
        generator.generate_dataset_in_independent_mode(num_samples, description_file)
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
