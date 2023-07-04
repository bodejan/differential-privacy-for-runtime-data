"""
    A simple dash interface for creating synthetic data for runtime prediction

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from scipy.stats import entropy, ks_2samp


import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_table



import cor_data_syn 
import ind_data_syn 
import eval_nn  


app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Synthetic Runtime Data"


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
        ds = cor_data_syn.CDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets/c3o-experiments/{dataset}.csv', dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {value1} from menu 1 and {value2}, {value3}, {value4}, {dataset} from menu 2.", f'assets/temp_{dataset}.png', n_clicks, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    else:
        ds = ind_data_syn.IDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets/c3o-experiments/{dataset}.csv', dataset=dataset)
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
        ds = eval_nn.NN(dataset, optimizer, split, epochs, n_clicks)
        return f"NN Trained", f'assets/normal{n_clicks}.png',f'assets/syn{n_clicks}.png', n_clicks


if __name__ == "__main__":
    app.run_server(port=8050)
