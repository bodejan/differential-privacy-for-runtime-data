from DataSynthesizer.DataGenerator import DataGenerator
import pandas as pd


class SDV_TVAE():
    def __init__(self,input_data:str,uuid:str, num_tuples: int = 1000, epochs: int=1500, batch_size: int=200):

        synthetic_data = f'dashboard/temp/{uuid}.csv'

        # Default config:
        configuration = {'enforce_min_max_values': True, 'enforce_rounding': True, 'epochs': epochs, 'batch_size': batch_size, 'compress_dims': [256, 256], 'decompress_dims': [256, 256], 'embedding_dim': 256, 'l2scale': 0.0001, 'loss_factor': 2}
        
        
        generator = DataGenerator()
        #generator.generate_dataset_in_independent_mode(num_tuples, description_file)
        generator.generate_dataset_sdv_tvae(num_tuples, input_data, configuration)
        generator.save_synthetic_data(synthetic_data)
        
        print("generator")

        # Read both datasets using Pandas.
        input_df = pd.read_csv(input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)

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
        fig.savefig(f'dashboard/assets/{uuid}_{dataset}.png')