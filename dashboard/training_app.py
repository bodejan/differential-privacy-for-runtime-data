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
                            html.Div([
                                html.Label('Epochs'),
                                dbc.Input(id="nn_epochs", type="number", value=10, min=10, max=100000, step=10),
                            ]),
                            html.Br(),
                            html.Div([
                                dcc.Loading(id="loading2", type="default",
                                            children=[
                                                dbc.Button('Submit', id='nnbutton', n_clicks=0,
                                                            style={'width': '100%'}),
                                                html.Div(id='outputnn')
                                            ])])
                        ], style={'margin': "0.5cm"}))
                ], width=4),
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Evaluation Metrics"),
                            dash_table.DataTable(id='nn_metrics', data=[], page_size=10),
                            html.Br(),
                            html.H4("Prediction Quality:"),
                            dcc.Graph(id="eval_fig"),
                            dcc.Graph(id="eval_fig2"),
                        ])),
                ], width=8),
            ], style={'height': '15cm'})])
    return layout


def training_callbacks(app):
    @app.callback(
        [dash.dependencies.Output('outputnn', 'children'),
         dash.dependencies.Output('eval_fig', 'figure'),
         dash.dependencies.Output('eval_fig2', 'figure'),
         dash.dependencies.Output('nn_metrics', 'data')
         ],
        [dash.dependencies.Input('nnbutton', 'n_clicks')],
        [dash.dependencies.State('dataset', 'value'),
         dash.dependencies.State('optimizer', 'value'),
         dash.dependencies.State('split', 'value'),
         dash.dependencies.State('nn_epochs', 'value'),
         dash.dependencies.State('session-id', 'children'),

         ], prevent_initial_call=True)
    def update_output(n_clicks, dataset, optimizer, split, epochs, uuid):
        print(n_clicks)
        if n_clicks == 0:
            return "Please enter a value and click the submit button.", n_clicks
        else:
            nn = eval_nn.NN()
            fig1, fig2, mse_original, mse_synthetic, mae_original, mae_synthetic, mape_original, mape_synthetic = nn.train(
                dataset, uuid[0], optimizer, split, epochs)

            data = [
                {'Name': 'MSE', 'Original': mse_original, 'Synthetic': mse_synthetic},
                {'Name': 'MAE', 'Original': mae_original, 'Synthetic': mae_synthetic},
                {'Name': 'MAPE', 'Original': mape_original, 'Synthetic': mape_synthetic},
            ]
            return f"NN Trained", fig1, fig2, data
