import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from scipy.stats import entropy, ks_2samp
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


class NN():
    """
    Class containing the neural network
    """
    def train(self, dataset, uuid, optimizer, split: int = 70, epochs: int = 10):
        """

        :param dataset: Dataset that was used for creating the synthetic data
        :param uuid: UUID to get access to the created synthetic data
        :param optimizer: Optimizer that should be used for training the NN
        :param split: Percentage of data used for training
        :param epochs: Epochs used in training
        :return: Multiple figures and metrics for evaluation of the performance of the trained NN
        """

        # Read in necessary files
        df = pd.read_csv(f"../datasets/{dataset}.csv")
        syndf = pd.read_csv(f'../dashboard/temp/{uuid}.csv')

        # for now, ignore machine type in training
        df = df.drop(["machine_type" ], axis=1)
        syndf = syndf.drop(["machine_type" ], axis=1)

        # get values from dataframe
        dataset = df.values
        syndataset = syndf.values

        # split the dataset rows
        X = dataset[:,0:7]
        Y = dataset[:,-1]
        synX = syndataset[:,0:7]
        synY = syndataset[:,-1]

        # conduct preprocessing on normal and synthetic data
        x_scaler = preprocessing.StandardScaler()
        y_scaler = preprocessing.StandardScaler()
        synx_scaler = preprocessing.StandardScaler()
        syny_scaler = preprocessing.StandardScaler()
        
        # make train test split from normal data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-(0.01*split))


        # fit the scaler for the normal data for preprocessing
        X = x_scaler.fit_transform(X_train)
        Y= y_scaler.fit_transform(Y_train.reshape(-1, 1) )

        # fit the scaler for the synthetic data for preprocessing
        synX = synx_scaler.fit_transform(synX)
        synY= syny_scaler.fit_transform(synY.reshape(-1, 1) )


        #create two simple models to train and predict with
        model = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])
        synmodel = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])

        # compile models
        model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
        synmodel.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])

        # fit models
        _ = model.fit(X, Y,
          batch_size=32, epochs=epochs)
        _ = synmodel.fit(synX, synY,
          batch_size=32, epochs=epochs)

        # predict the runtime for samples in test dataset
        pred = model.predict(x_scaler.transform(X_test))
        synpred = synmodel.predict(synx_scaler.transform(X_test))

        # scale runtime predictions back in to normal seconds
        scaled_y =y_scaler.inverse_transform(pred) 
        synscaled_y =y_scaler.inverse_transform(synpred)


        def calculate_mse(prediction, truth):
            """
            Function to calculate mean squared error
            :param prediction: Array with predicted values
            :param truth: Array with the true values
            :return: MSE
            """
            mse = 0
            for i in range(len(prediction)):
                mse += (truth[i]-prediction[i])**2
            return mse


        def calculate_mae(prediction, truth):
            """
            Function to calculate mean absolut error
            :param prediction: Array with predicted values
            :param truth: Array with true values
            :return: MAE
            """
            mae = 0
            for i in range(len(prediction)):
                mae += abs(truth[i] - prediction[i])
            return mae

        def calculate_mape(prediction, truth):
            """
            Function to calculate mean absolut percentage error
            :param prediction: Array with predicted values
            :param truth: Array with true values
            :return: MAPE
            """
            mape = 0
            for i in range(len(prediction)):
                mape += np.mean(np.abs((truth[i] - prediction[i]) / truth[i])) * 100
            return mape


        def plot_scatter(x_values, y_values, y_values_2, stri, s1, s2):
            """
            Create Scatte plot to visualize difference between predicted and true values

            :param x_values:
            :param y_values:
            :param y_values_2:
            :param stri:
            :param s1:
            :param s2:
            :return:
            """
            fig, ax = plt.subplots()

            # Plot the scatter plot
            ax.scatter(x_values, y_values, label='Predicted')
            ax.scatter(x_values, y_values_2, label='Actual')

            ax.set_xlabel('Datasample')
            ax.set_ylabel('Runtime in ms')
            ax.legend()

            if stri == "syn":
                ax.set_title(f'Neural Network trained with {s2} Synthetic Data')
            else: 
                ax.set_title(f'Neural Network trained with {s1} Private Data')

            return fig

        def plot_scatter2(y_values, y_values_2, stri, s1, s2):
            # Create a scatter plot
            x_values = list(range(1, len(y_values) + 1))

            # Create figure with secondary y-axis
            df =  {'x': x_values, 'Predicted':y_values.flatten() , 'True Labels': y_values_2.flatten()}
            fig = px.scatter(df, x='x', y=['Predicted', 'True Labels'])

            fig.update_layout(
                xaxis=dict(title='Datasample'),
                yaxis=dict(title='Runtime in ms'),
                legend_title="Legend Title",
                title=f'Neural Network trained with {s2} Synthetic Data' if stri == 'syn' else f'Neural Network trained with {s1} Private Data',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )

            return fig

        return plot_scatter2(scaled_y, Y_test, "normal", X.shape, synX.shape), plot_scatter2(synscaled_y, Y_test, "syn",X.shape, synX.shape), calculate_mse(scaled_y, Y_test), calculate_mse(synscaled_y, Y_test), calculate_mae(scaled_y, Y_test), calculate_mae(
                synscaled_y, Y_test), calculate_mape(scaled_y, Y_test), calculate_mape(synscaled_y, Y_test)

