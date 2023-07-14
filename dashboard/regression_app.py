
import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

import eval_reg
import pandas as pd


def regression_layout():
    layout = html.Div(
    [
        html.Br(),
        dbc.Row([
        dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H4("Evaluation using Regression Methods"),


                html.Div([
                    dcc.Loading(id="loading3", type="circle",
                                children=[
                                    html.Button('Submit', id='regbutton', n_clicks=0,
                                                style={'align': 'center', 'width': '100%', 'display': 'inline-block',
                                                       'background-color': '#4CAF50', 'color': 'white'}),
                                    html.Div(id='outputreg')
                                ])])
            ]))], width=4),
    dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H4("Prediction Quality:"),
                html.Div(id="results"),
            ])),
    ], width=8),
    ], style={'height' :'15cm'})])
    return layout


def regression_callbacks(app):
    @app.callback(
        [dash.dependencies.Output('outputreg', 'children'),
         ],
        [dash.dependencies.Input('regbutton', 'n_clicks')],
        [dash.dependencies.State('dataset', 'value'),
         dash.dependencies.State('session-id', 'children'),
         ], prevent_initial_call=True)
    def update_output(n_clicks, dataset, uuid):
        print("Regression")
        return ["Result =" + str(eval_reg.Regression.eval_input_data('../datasets/sort.csv'))]