import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SDV_TVAE():
    def request(self,input_data:str,uuid:str, num_tuples: int = 1000, epochs: int=1500, batch_size: int=200):

        synthetic_data = f'dashboard/temp/{uuid}.csv'
        # Read original dataset.
        input_df = pd.read_csv(input_data)

        # Default config:
        configuration = {'enforce_min_max_values': True, 'enforce_rounding': True, 'epochs': epochs, 'batch_size': batch_size, 'compress_dims': [256, 256], 'decompress_dims': [256, 256], 'embedding_dim': 256, 'l2scale': 0.0001, 'loss_factor': 2}
        
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

        synthesizer.fit(input_df)
        synthetic_df = synthesizer.sample(num_tuples)
        synthetic_df.to_csv(synthetic_data, index=False)

        quality_report = evaluate_quality(
            real_data=input_df,
            synthetic_data=synthetic_df,
            metadata=metadata
        )

        col_shapes_plt = quality_report.get_visualization(property_name='Column Shapes')
        col_pair_trends_plt = quality_report.get_visualization(property_name='Column Pair Trends')

        # Display the combined plot
        return col_shapes_plt, col_pair_trends_plt

if __name__ == "__main__":
    sdv_tvae= SDV_TVAE()
    col_shapes_plt, col_pair_trends_plt = sdv_tvae.request('datasets/sort.csv', 'sort_synthetic_123') 
    col_shapes_plt.show()
    col_pair_trends_plt.show()
