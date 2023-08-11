import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

from synthesizer import Synthesizer


class SDVSynthesizer(Synthesizer):
    """
    Synthesizer using the SDV package to generate synthetic data using the TVAESynthesizer.

    Args:
        input_data (str): Path to the input CSV file containing the original dataset.
        session_id (str): Identifier for the session.

    Methods:
        request(num_tuples, epochs, batch_size, enforce_min_max_values, compress_dims, decompress_dims, **_):
            Generate synthetic data using the TVAE model and evaluate its quality.
    """
    def request(self, num_tuples: int = 1000, epochs: int = 1500, batch_size: int = 200,
                enforce_min_max_values=True, compress_dims=None, decompress_dims=None, **_):
        """
        Generate synthetic data using the TVAE model and evaluate its quality.

        Args:
            num_tuples (int, optional): Number of synthetic tuples to generate (default: 1000).
            epochs (int, optional): Number of epochs for training (default: 1500).
            batch_size (int, optional): Batch size for training (default: 200).
            enforce_min_max_values (bool, optional): Enforce minimum and maximum values (default: True).
            compress_dims (int, optional): Number of dimensions for data compression (default: None).
            decompress_dims (int, optional): Number of dimensions for data decompression (default: None).

        Returns:
            plotly.graph_objs._figure.Figure: Visualization of column shapes.
            plotly.graph_objs._figure.Figure: Visualization of column pair trends.
        """
        if compress_dims is None:
            compress_dims = [256, 256]
        else:
            compress_dims = [int(compress_dims), int(compress_dims)]

        if decompress_dims is None:
            decompress_dims = [256, 256]
        else:
            decompress_dims = [int(decompress_dims), int(decompress_dims)]
        synthetic_data = f'temp/{self._session_id}.csv'
        
        # Read original dataset.
        input_df = pd.read_csv(self._input_data)
        
        # Default config:
        configuration = {'enforce_min_max_values': enforce_min_max_values, 'enforce_rounding': True,
                         'epochs': epochs, 'batch_size': batch_size,
                         'compress_dims': compress_dims, 'decompress_dims': decompress_dims,
                         'embedding_dim': 256, 'l2scale': 0.0001, 'loss_factor': 2}

        print('TVAE configuration: ', configuration)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=input_df)
        metadata.validate()

        synthesizer = TVAESynthesizer(
            metadata=metadata,
            enforce_min_max_values=configuration['enforce_min_max_values'],
            epochs=configuration['epochs'],
            batch_size=configuration['batch_size'],
            compress_dims=configuration['compress_dims'],
            decompress_dims=configuration['decompress_dims'],
            embedding_dim=configuration['embedding_dim'],
            l2scale=configuration['l2scale'],
            loss_factor=configuration['loss_factor']
        )
        print("fit sdv")

        synthesizer.fit(input_df)
        print("fitted sdv")

        synthetic_df = synthesizer.sample(num_tuples)
        synthetic_df.to_csv(synthetic_data, index=False)

        quality_report = evaluate_quality(
            real_data=input_df,
            synthetic_data=synthetic_df,
            metadata=metadata
        )

        col_shapes_plt = quality_report.get_visualization(property_name='Column Shapes')
        col_pair_trends_plt = quality_report.get_visualization(property_name='Column Pair Trends')

        return col_shapes_plt, col_pair_trends_plt


if __name__ == "__main__":
    sdv_tvae = SDVSynthesizer('datasets/sort.csv', 'sort_synthetic_123')
    col_shapes_plt, col_pair_trends_plt = sdv_tvae.request()
    col_shapes_plt.show()
    col_pair_trends_plt.show()
