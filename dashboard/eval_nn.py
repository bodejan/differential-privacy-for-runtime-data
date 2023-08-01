import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

import plotly.express as px

import numpy as np


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


        def plot_scatter(predicted_values, true_values, mode: str="syn", num_private: int = 0, num_synthetic: int = 0):
            """
            Create Scatter plot to visualize difference between predicted and true values

            :param predicted_values: Predicted Runtimes
            :param true_values: True runtimes
            :param mode: Should the plot show synthetic or private data
            :param num_private: Number of private Samples to train the model
            :param num_synthetic: Number of Normal Samples to train the model
            :return: Figure of scatter plot
            """
            x_values = list(range(1, len(predicted_values) + 1))

            df =  {'x': x_values, 'Predicted':predicted_values.flatten() , 'True Labels': true_values.flatten()}
            fig = px.scatter(df, x='x', y=['Predicted', 'True Labels'])

            fig.update_layout(
                xaxis=dict(title='Datasample'),
                yaxis=dict(title='Runtime in ms'),
                legend_title="Legend Title",
                title=f'Neural Network trained with {num_synthetic} Synthetic Data' if mode == 'syn' else f'Neural Network trained with {num_private} Private Data',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )

            return fig

        return plot_scatter(scaled_y, Y_test, "normal", X.shape, synX.shape), plot_scatter(synscaled_y, Y_test, "syn",X.shape, synX.shape), calculate_mse(scaled_y, Y_test), calculate_mse(synscaled_y, Y_test), calculate_mae(scaled_y, Y_test), calculate_mae(
                synscaled_y, Y_test), calculate_mape(scaled_y, Y_test), calculate_mape(synscaled_y, Y_test)

