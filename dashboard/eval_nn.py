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



class NN():
    def train(self, dataset, optimizer, split: int = 70, epochs: int = 10, n_clicks: int = 0):
        print("Train Model")
        # Read the given CSV file, and view some sample records
        df = pd.read_csv(f"datasets/{dataset}.csv")
        syndf = pd.read_csv("dashboard/temp/sythetic_data.csv")
        df = df.drop(["machine_type" ], axis=1)
        syndf = syndf.drop(["machine_type" ], axis=1)
        dataset = df.values
        syndataset = syndf.values

        X = dataset[:,0:7]
        Y = dataset[:,-1]
        synX = syndataset[:,0:7]
        synY = syndataset[:,-1]

        x_scaler = preprocessing.StandardScaler()
        y_scaler = preprocessing.StandardScaler()
        synx_scaler = preprocessing.StandardScaler()
        syny_scaler = preprocessing.StandardScaler()
        
        print("finished standard Model")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-(0.01*split))


        X = x_scaler.fit_transform(X_train)
        Y= y_scaler.fit_transform(Y_train.reshape(-1, 1) )
        
        synX = synx_scaler.fit_transform(synX)
        synY= syny_scaler.fit_transform(synY.reshape(-1, 1) )
        
        model = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])
        synmodel = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])
        print("Compiled standard Model")

        model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
        synmodel.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])

        hist = model.fit(X, Y,
          batch_size=32, epochs=epochs)
        hist = synmodel.fit(synX, synY,
          batch_size=32, epochs=epochs)


        pred = model.predict(x_scaler.transform(X_test))
        synpred =synmodel.predict(synx_scaler.transform(X_test))

        scaled_y =y_scaler.inverse_transform(pred) 
        synscaled_y =y_scaler.inverse_transform(synpred)

        def calculate_mse(prediction, truth):
            mse = 0
            for i in range(len(prediction)):
                mse += (truth[i]-prediction[i])**2
            return mse


        def calculate_mae(prediction, truth):
            mae = 0
            for i in range(len(prediction)):
                mae += abs(truth[i] - prediction[i])
            return mae


        def plot_scatter(x_values, y_values, y_values_2, stri, s1, s2):

            # Create a figure and axis
            fig, ax = plt.subplots()        

            # Plot the scatter plot
            ax.scatter(x_values, y_values, label='Predicted')
            ax.scatter(x_values, y_values_2, label='Actual')


            # Set labels for x and y axes
            ax.set_xlabel('Datasample')
            ax.set_ylabel('Runtime in ms')
            ax.legend()
            # Set a title for the plot
            if stri == "syn":
                ax.set_title(f'Neural Network trained with {s2} Synthetic Data')
            else: 
                ax.set_title(f'Neural Network trained with {s1} Private Data')

            # Show the plot
            return fig
            #fig.savefig(f'dashboard/assets/{stri}{n_clicks}.png')

        def plot_scatter2(y_values, y_values_2, stri, s1, s2):
            # Create a scatter plot
            fig = go.Figure()
            x_values = list(range(1, len(y_values) + 1))

            # Add the scatter traces for predicted and actual values
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Predicted'))
            fig.add_trace(go.Scatter(x=x_values, y=y_values_2, mode='markers', name='Actual'))
                  # Set the layout of the plot
            fig.update_layout(
                xaxis=dict(title='Datasample'),
                yaxis=dict(title='Runtime in ms'),
                title=f'Neural Network trained with {s2} Synthetic Data' if stri == 'syn' else f'Neural Network trained with {s1} Private Data',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )

            return fig
        return plot_scatter2(scaled_y, Y_test, "normal", X.shape, synX.shape), plot_scatter2(synscaled_y, Y_test, "syn",X.shape, synX.shape), calculate_mse(scaled_y, Y_test), calculate_mse(synscaled_y, Y_test), calculate_mae(scaled_y, Y_test), calculate_mae(
                synscaled_y, Y_test)

