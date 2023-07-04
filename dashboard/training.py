
import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_table
import eval_nn
import pandas as pd


def training_layout():
    layout = html.Div(
    [
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
                    dcc.Loading(id="loading2" ,type="circle",
                                children=[
                                    html.Button('Submit', id='nnbutton', n_clicks=0, style={'align': 'center', 'width' :'100%', 'display': 'inline-block', 'background-color': '#4CAF50', 'color': 'white'}),
                                    html.Div(id='outputnn')
                                ])])
            ], style={'margin' :"0.5cm"}))
    ], width=4),
    dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H4("Prediction Quality:"),
                html.Img(id="eval_imagenn", src='assets/no.png', style={'height' :'14cm'}),
                html.Img(id="eval_imagenn2", src='assets/no.png', style={'height' :'14cm'}),

            ])),
    ], width=8),
    ], style={'height' :'15cm'})])
    return layout

def training_callbacks(app):
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
         ], prevent_initial_call=True)
    def update_output(n_clicks, dataset, optimizer, split, epochs, current_src):
        print(n_clicks)
        if n_clicks == 0:
            return "Please enter a value and click the submit button.", current_src, n_clicks
        else:
            ds = eval_nn.NN(dataset, optimizer, split, epochs, n_clicks)
            return f"NN Trained", f'assets/normal{n_clicks}.png', f'assets/syn{n_clicks}.png', n_clicks
