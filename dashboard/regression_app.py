
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
                html.H4("Training of NN"),
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
