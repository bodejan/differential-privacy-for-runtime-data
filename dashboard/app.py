"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""
import logging
import threading

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D
import dash_table

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import dash
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State
from dash import html
from sklearn import preprocessing


from scipy.stats import entropy, ks_2samp

from DataSynthesizer.lib.utils import pairwise_attributes_mutual_information, normalize_given_distribution

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd


global cds 
global ids 


class CDS():
    def __init__(self, epsilon: int = 0, num_tuples: int = 1000, input_data:str = "../datasets/c3o-experiments/sort.csv", dataset: str="sort"):

        description_file = f'temp/description.json'
        synthetic_data = f'temp/sythetic_data.csv'

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


class IDS():
    def __init__(self, epsilon: int = 0, num_tuples: int = 1000, input_data:str = "../datasets/c3o-experiments/sort.csv", dataset: str="sort"):

        description_file = f'temp/description.json'
        synthetic_data = f'temp/sythetic_data.csv'

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
        fig.savefig(f'assets/temp_ids_{dataset}.png')





log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


#DO implement continious evaluation metrics receiving from clients, save the values and adapt callbacks to plot them
app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Synthetic Runtime Data"
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9]
fpr = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
tpr = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
acc = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
client_list = [["client_2", 58666, "up", 0, 0, 0, 0], ["client_5", 58999, "up", 0, 0, 0, 0]]
train_c5 = False
train_c2 = False

app.layout = html.Div(
    [
        html.Br(),
        dbc.Card(
            dbc.CardBody([
                html.H1("Synthetic Data for Runtime Prediction", style={'align':"center"}),
            ]), style={'margin':"0.5cm"}, color="primary"),
        html.Br(),
        dbc.Row([
            dbc.Col([
            dbc.Card(
                dbc.CardBody([
                            html.H4("Synthetisation"),
                            html.Br(),
                            html.Div([
                                html.Label('Select Synthesizer'),
                                dcc.Dropdown(
                                    id='dropdown1',
                                    options=[
                                        {'label': 'SDV TVAE', 'value': 'sdv'},
                                        {'label': 'Correlated DS', 'value': 'cds'},
                                        {'label': 'Independent DS', 'value': 'ids'}
                                    ],
                                    value='ids'
                                )
                            ]),
                            html.Br(),
                            html.Div([
                                html.Label('Select Dataset'),
                                dcc.Dropdown(
                                    id='dropdown2',
                                    options=[
                                        {'label': 'C3O Kmeans', 'value': 'kmeans'},
                                        {'label': 'C3O Sort', 'value': 'sort'},
                                        {'label': 'C3O Grep', 'value': 'grep'},
                                        {'label': 'C3O SGD', 'value': 'sgd'},
                                        {'label': 'C3O Pagerank', 'value': 'pagerank'},
                                    ],
                                    value=''
                                )
                            ]),
                            html.Br(),
                            html.P("Select Epsilon"),
                            dcc.Slider(0, 1, 0.1, value=0, id='epsilon'),
                            html.Br(),
                            html.P("Amount of Data to Generate"),
                            dbc.Input(id = "num", type="number", value = 1000, min=10, max=100000, step=10),
                            html.Br(),
                            html.Div([                            
                                dcc.Loading(id="loading",type="circle", 
                                children=[
                                html.Button('Submit', id='submit-button', n_clicks=0, style={'align': 'center', 'width':'100%', 'display': 'inline-block', 'background-color': '#4CAF50', 'color': 'white'}),
                                html.Div(id='output')
                            ])])

                ], style={'margin':"0.5cm", 'height':'100%'}))
                ], width=4),
            dbc.Col([
                dbc.Card(
                dbc.CardBody([
                    html.H4("Evaluation of Synthetic Data"),
                    html.P("Selected Dataset:"),
                    dash_table.DataTable(id='csv-table', data="", columns="", page_size=10),
                    html.P("Generated Synthetic Data:"),
                    dash_table.DataTable(id='csv-table1', data="", columns="", page_size=10),
                    html.Br(),
                    html.Img(id="eval_image", src='assets/no.png', style={'height':'14cm'}),
                    ], style={'height':'100%'}))
                    ], width=8),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                dbc.CardBody([
                            html.H4("Training of NN"),
                            html.Label('Select Dataset'),
                                dcc.Dropdown(
                                    id='datasetnn',
                                    options=[
                                        {'label': 'C3O Kmeans', 'value': 'kmeans'},
                                        {'label': 'C3O Sort', 'value': 'sort'},
                                        {'label': 'C3O Grep', 'value': 'grep'},
                                        {'label': 'C3O SGD', 'value': 'sgd'},
                                        {'label': 'C3O Pagerank', 'value': 'pagerank'},
                                    ],
                                    value=''
                                ),
                            html.Div([
                                html.Label('Select Optimizer'),
                                dcc.Dropdown(
                                    id='dropdown3',
                                    options=[
                                        {'label': 'RMSprop', 'value': 'RMSprop'},
                                        {'label': 'Adagrad', 'value': 'Adagrad'},
                                        {'label': 'Adam', 'value': 'adam'},
                                        {'label': 'SGD', 'value': 'sgd'}
                                    ],
                                    value='adam'
                                )
                            ]),
                            html.P("Select Test/Train Split"),
                            dcc.Slider(30, 90, 5, value=70, id='split'),
                            html.P("Select Epochs"), 
                            dcc.Slider(1, 90, 5, value=20, id='epochs'), 
                            html.Br(),
                            html.Div([                            
                                dcc.Loading(id="loading2",type="circle", 
                                children=[
                                html.Button('Submit', id='nnbutton', n_clicks=0, style={'align': 'center', 'width':'100%', 'display': 'inline-block', 'background-color': '#4CAF50', 'color': 'white'}),
                                html.Div(id='outputnn')
                            ])])
                    ], style={'margin':"0.5cm"}))
            ], width=4),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Prediction Quality:"),
                        html.Img(id="eval_imagenn", src='assets/no.png', style={'height':'14cm'}),
                        html.Img(id="eval_imagenn2", src='assets/no.png', style={'height':'14cm'}),

                    ])),
            ], width=8),
        ], style={'height':'15cm'}),
    ])

@app.callback(
    [dash.dependencies.Output('csv-table', 'data'), 
    dash.dependencies.Output('csv-table', 'columns')],
    [dash.dependencies.Input('dropdown2', 'value')],
    [dash.dependencies.State('dropdown2', 'value')],
    prevent_initial_call=True)
def update_output(value, v2):
    df = pd.read_csv(f'../datasets/c3o-experiments/{value}.csv')
    return df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]


@app.callback(
    [dash.dependencies.Output('output', 'children'),
    dash.dependencies.Output('eval_image', 'src'), 
    dash.dependencies.Output('submit-button', 'n_clicks'),
    dash.dependencies.Output('csv-table1', 'data'), 
    dash.dependencies.Output('csv-table1', 'columns')],
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('dropdown1', 'value'),
     dash.dependencies.State('epsilon', 'value'),
     dash.dependencies.State('split', 'value'),
     dash.dependencies.State('num', 'value'),
     dash.dependencies.State('dropdown2', 'value'),
     dash.dependencies.State('eval_image', 'src')
     ],prevent_initial_call=True)
def update_output(n_clicks, value1, value2, value3, value4, dataset, current_src):
    print(n_clicks)
    if n_clicks == 0 :
        return "Please enter a value and click the submit button.", current_src, n_clicks
    elif value4 is None or dataset =="":
        return "Input value is required!", current_src, n_clicks, "", ""
    elif value1 == "cds":
        ds = CDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets/c3o-experiments/{dataset}.csv', dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {value1} from menu 1 and {value2}, {value3}, {value4}, {dataset} from menu 2.", f'assets/temp_{dataset}.png', n_clicks, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    else:
        ds = IDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets/c3o-experiments/{dataset}.csv', dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {value1} from menu 1 and {value2}, {value3}, {value4}, {dataset} from menu 2.", f'assets/temp_ids_{dataset}.png', n_clicks, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]

@app.callback(
    [dash.dependencies.Output('outputnn', 'children'),
    dash.dependencies.Output('eval_imagenn', 'src'), 
    dash.dependencies.Output('eval_imagenn2', 'src'), 
    dash.dependencies.Output('nnbutton', 'n_clicks')],
    [dash.dependencies.Input('nnbutton', 'n_clicks')],
    [dash.dependencies.State('datasetnn', 'value'),
     dash.dependencies.State('dropdown3', 'value'),
     dash.dependencies.State('split', 'value'),
     dash.dependencies.State('epochs', 'value'),
     dash.dependencies.State('eval_imagenn', 'src')
     ],prevent_initial_call=True)
def update_output(n_clicks, dataset, optimizer, split, epochs, current_src):
    print(n_clicks)
    if n_clicks == 0 :
        return "Please enter a value and click the submit button.", current_src, n_clicks
    else:
        ds = NN(dataset, optimizer, split, epochs, n_clicks)
        return f"NN Trained", f'assets/normal{n_clicks}.png',f'assets/syn{n_clicks}.png', n_clicks


class NN(): 
    def __init__(self, dataset, optimizer, split: int = 70, epochs: int = 10, n_clicks: int = 0):
        print("Train Model")
        # Read the given CSV file, and view some sample records
        df = pd.read_csv(f"../datasets/c3o-experiments/{dataset}.csv")
        syndf = pd.read_csv("../dahboard/temp/sythetic_data.csv")
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
        
        def plot_scatter(x_values, y_values, y_values_2, stri, s1, s2):

            # Create a figure and axis
            fig, ax = plt.subplots()        
            x_values = list(range(1, len(y_values) + 1))

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
            fig.savefig(f'assets/{stri}{n_clicks}.png')
        
        plot_scatter(scaled_y, scaled_y, Y_test, "normal", X.shape, synX.shape)
        plot_scatter(synscaled_y, synscaled_y, Y_test, "syn",X.shape, synX.shape)



if __name__ == "__main__":
    app.run_server(port=8050)
