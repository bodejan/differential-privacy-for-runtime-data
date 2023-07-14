
import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

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

                html.Div([
                    html.Label('Select Optimizer'),
                    dcc.Dropdown(
                        id='optimizer',
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
                dcc.Graph(id="eval_fig"),
                dcc.Graph(id="eval_fig2"),
                html.Div(id="metrics"),
            ])),
    ], width=8),
    ], style={'height' :'15cm'})])
    return layout

def training_callbacks(app):
    @app.callback(
        [dash.dependencies.Output('outputnn', 'children'),
         dash.dependencies.Output('eval_fig', 'figure'),
         dash.dependencies.Output('eval_fig2', 'figure'),
         dash.dependencies.Output('nnbutton', 'n_clicks'), dash.dependencies.Output('metrics', 'children')
         ],
        [dash.dependencies.Input('nnbutton', 'n_clicks')],
        [dash.dependencies.State('dataset', 'value'),
         dash.dependencies.State('optimizer', 'value'),
         dash.dependencies.State('split', 'value'),
         dash.dependencies.State('epochs', 'value'),
         dash.dependencies.State('session-id', 'children'),

         ], prevent_initial_call=True)
    def update_output(n_clicks, dataset, optimizer, split, epochs, uuid):
        print(n_clicks)
        if n_clicks == 0:
            return "Please enter a value and click the submit button.", n_clicks
        else:
            nn = eval_nn.NN()
            fig1, fig2, mse1, mse2, mse3, mse4, mse5, mse6 = nn.train(dataset, uuid[0], optimizer, split, epochs, n_clicks)
            return f"NN Trained", fig1, fig2, n_clicks, f'MSE (original): {mse1}, MSE (synthetic):{mse2}, MAE(original):{mse3}, MAE (synthetic): {mse4}, MAPE (original): {mse5}, MAPE (synthetic): {mse6}'
