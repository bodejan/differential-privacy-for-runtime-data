from typing import Union

import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

import eval_reg
import pandas as pd

from meta_information import MetaInformation

Results = list[dict[str, Union[str, float]]]

columns = [
    {'id': 'name', 'name': 'Name'},
    {'id': 'std', 'name': 'Standard deviation'},
    {'id': 'mse', 'name': 'Mean squared error'},
    {'id': 'mape', 'name': 'Mean absolute percentage error'}
]


def regression_layout():
    """
    Defines the layout of the regression evaluation dashboard.

    Returns:
        layout (dash.html.Div): The layout of the dashboard.
    """
    layout = html.Div(
        [
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Evaluation using Regression Methods"),
                            html.Br(),
                            html.P("Select Test/Train Split"),
                            dcc.Slider(30, 90, 5, value=70, id='reg_split'),
                            html.Br(),
                            html.Div([
                                dcc.Loading(id="loading-5", type="default",
                                            children=[
                                                dbc.Button('Submit', id='regbutton', n_clicks=0,
                                                           style={'width': '100%'}),
                                                html.Div(id='outputreg')
                                            ])])
                        ]))], width=4),
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Evaluation metrics (original data):"),
                            dash_table.DataTable(id='regression_results_original', data=[], columns=columns,
                                                 page_size=10),
                            html.Br(),
                            html.H4("Evaluation metrics (synthetic data):"),
                            dash_table.DataTable(id='regression_results_synthetic', data=[], columns=columns,
                                                 page_size=10),
                        ])),
                ], width=8),
            ], style={'height': '15cm'})])
    return layout


def postprocess_results(results: Results) -> Results:
    """
    Post-processes the evaluation results to round float values to two decimal places.

    Args:
        results (Results): List of evaluation results.

    Returns:
        Results: Post-processed evaluation results.
    """
    return [
        {key: round(value, 2) if isinstance(value, float) else value for key, value in result.items()} for result in
        results
    ]


def regression_callbacks(app):
    """
    Defines the callback functions for the regression evaluation dashboard.

    Args:
        app (dash.Dash): The Dash app instance.
    """
    @app.callback(
        [
            dash.dependencies.Output('regression_results_original', 'data'),
            dash.dependencies.Output('regression_results_synthetic', 'data'),
            dash.dependencies.Output('outputreg', 'children'),
        ],
        [dash.dependencies.Input('regbutton', 'n_clicks'), dash.dependencies.State('reg_split', 'value') ],
        [dash.dependencies.State('session-id', 'children')],
        prevent_initial_call=True
    )
    def update_output(n_clicks, split, uuid: list[str]):
        """
        Updates the displayed results based on user inputs.

        Args:
            n_clicks (int): Number of times the submit button is clicked.
            split (float): Test/train split percentage.
            uuid (list[str]): List containing the session ID.

        Returns:
            list: Updated data for DataTables and an empty string for outputreg.
        """
        session_id = uuid[0]
        meta = MetaInformation.from_id_file(session_id)
        print("Regression")

        original_results = postprocess_results(
            eval_reg.Regression.eval_input_data(input_file=f'../datasets/{meta.dataset_name}.csv',
                                                dataset_name=meta.dataset_name, split=split/100)
        )
        synthetic_results = postprocess_results(
            eval_reg.Regression.eval_input_data(input_file=f'temp/{session_id}.csv', dataset_name=meta.dataset_name,
                                                 split=split/100, original_file=f'../datasets/{meta.dataset_name}.csv')
        )

        return [
            original_results,
            synthetic_results,
            ''
        ]
