"""
    A simple dash interface for creating synthetic data for runtime prediction

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""

import uuid

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, dash_table
from dash.dependencies import Output, Input, State

import cor_data_syn
import ind_data_syn
import sdv_tvae_syn
from meta_information import MetaInformation
from regression_app import regression_layout, regression_callbacks
from synthesizer import Synthesizer
from training_app import training_callbacks, training_layout

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
)
app.title = "Synthetic Runtime Data"

navbar = html.Div(
    html.H1('Synthetic Data for Runtime Prediction', style={'color': 'white', 'margin': '0'}),
    style={
        'background-color': '#0b1120',
        "padding": "16px 32px",
        "position": 'sticky'
    },
)


def session_id():
    return str(uuid.uuid4()),


home_content = dbc.Row([
    html.Br(),
    dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H4("Synthetisation"),
                html.Br(),
                html.Div([
                    html.Label('Select Synthesizer'),
                    dcc.Dropdown(
                        id='synthesizer',
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
                        id='dataset',
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
                html.Div(id='specific_options'),
                html.Br(),
                html.P("Amount of Data to Generate"),
                dbc.Input(id="amount", type="number", value=1000, min=10, max=100000, step=10),
                html.Br(),
                html.Div([
                    dcc.Loading(id="loading", type="circle",
                                children=[
                                    html.Button('Create', id='create-button', n_clicks=0,
                                                style={'align': 'center', 'width': '100%', 'display': 'inline-block',
                                                       'background-color': '#4CAF50', 'color': 'white'}),
                                    html.Div(id='output')
                                ])]),
                html.Div([
                    dcc.Loading(id="loadingdownload", type="circle",
                                children=[
                                    html.Button('Download File', id='btn-download-txt', n_clicks=0,
                                                style={'align': 'center', 'width': '100%', 'display': 'inline-block',
                                                       'background-color': '#4CAF50', 'color': 'white'}),
                                    html.Div(id='output-download')
                                ])]),
                dcc.Download(id="download-text")

            ], style={'marginLeft': "0.5cm", 'height': '100%'}))
    ], width=4),
    dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H4("Evaluation of Synthetic Data"),
                html.P("Selected Dataset:"),
                dash_table.DataTable(id='csv-table-original', data=[], columns=[], page_size=10),
                html.P("Generated Synthetic Data:"),
                dash_table.DataTable(id='csv-table-synthetic', data=[], columns=[], page_size=10),
                html.Br(),
                dcc.Graph(id="eval_image"),
                dcc.Graph(id="eval_image2"),
                dcc.Graph(id="eval_image3"),

            ], style={'height': '100%'}))
    ], width=8),
])

app.layout = html.Div(
    [
        navbar,
        html.Div(session_id(), id='session-id'),  # style={'display': 'none'}),
        html.Br(),
        dbc.Tabs(
            [
                dbc.Tab(home_content, label='Data Creation'),
                dbc.Tab(training_layout(), label='Runtime Prediction with NN'),
                dbc.Tab(regression_layout(), label='Runtime Prediction with Regression Models')
            ]
        )
    ])

training_callbacks(app)
regression_callbacks(app)


@app.callback(
    Output("download-text", "data"),
    State('session-id', 'children'),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)
def func(session_id, n_clicks):
    return dcc.send_file(
        f'temp/{session_id[0]}.csv'
    )


@app.callback(
    [dash.dependencies.Output('csv-table-original', 'data'),
     dash.dependencies.Output('csv-table-original', 'columns')],
    [dash.dependencies.Input('dataset', 'value')
     ],
    prevent_initial_call=True)
def update_output(value):
    df = pd.read_csv(f'../datasets/{value}.csv')
    return df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]


@app.callback(
    [dash.dependencies.Output('specific_options', 'children')],
    [dash.dependencies.Input('synthesizer', 'value')],
)
def show_synthesizer_options(synthesizer_name: str):
    print(synthesizer_name)
    if synthesizer_name == 'sdv':
        return [html.Div([
            html.Div([
                html.Label('Compress dims'),
                dcc.Dropdown(
                    id='compress_dims',
                    options=[
                        {'label': '64x64', 'value': '64'},
                        {'label': '128x128', 'value': '128'},
                        {'label': '256x256', 'value': '256'}
                    ],
                    value='256'
                )
            ]),
            html.Div([
                html.Label('decompress_dims'),
                dcc.Dropdown(
                    id='decompress_dims',
                    options=[
                        {'label': '64x64', 'value': '64'},
                        {'label': '128x128', 'value': '128'},
                        {'label': '256x256', 'value': '256'}
                    ],
                    value='256'
                )
            ]),
            html.Div([
                html.Label('Enforce min-max values'),
                dcc.Dropdown(
                    id='enforce_min_max_values',
                    options=[
                        {'label': 'True', 'value': True},
                        {'label': 'False', 'value': False},
                    ],
                    value='true'
                )
            ]),
            html.Div([
                html.Label('Epochs'),
                dbc.Input(id="epochs", type="number", value=10, min=10, max=100000, step=10),
            ]),
            html.Div([
                html.Label('Batch size'),
                dbc.Input(id="batch_size", type="number", value=200, min=100, max=1000, step=100),
            ]),

            dcc.Input(id='epsilon', style={'display': 'none'}),
        ])]
    else:
        return [html.Div([
            dcc.Input(id='enforce_min_max', style={'display': 'none'}),
            dcc.Input(id='decompress_dims', style={'display': 'none'}),
            dcc.Input(id='compress_dims', style={'display': 'none'}),
            dcc.Input(id='epochs', style={'display': 'none'}),
            dcc.Input(id='batch_size', style={'display': 'none'}),
            html.P("Select Epsilon"),
            dcc.Slider(0, 1, 0.1, value=0, id='epsilon'),
        ])]


@app.callback(
    [dash.dependencies.Output('output', 'children'),
     dash.dependencies.Output('eval_image', 'figure'),
     dash.dependencies.Output('eval_image2', 'figure'),
     dash.dependencies.Output('eval_image3', 'figure'),
     dash.dependencies.Output('csv-table-synthetic', 'data'),
     dash.dependencies.Output('csv-table-synthetic', 'columns')],
    dict(
        default_inputs=[
            dash.dependencies.State('synthesizer', 'value'),
            dash.dependencies.State('dataset', 'value'),
            dash.dependencies.State('epsilon', 'value'),
            dash.dependencies.State('amount', 'value'),
            dash.dependencies.State('session-id', 'children')
        ],
        sdv_options=dict(
            enforce_min_max_values=dash.dependencies.State('enforce_min_max_values', 'value'),
            compress_dims=dash.dependencies.State('compress_dims', 'value'),
            decompress_dims=dash.dependencies.State('decompress_dims', 'value'),
            epochs=dash.dependencies.State('epochs', 'value'),
            batch_size=dash.dependencies.State('batch_size', 'value'),
        ),
        n_clicks=dash.dependencies.Input('create-button', 'n_clicks')
    )
    , prevent_initial_call=True)
def update_output(default_inputs, sdv_options, n_clicks):
    synthesizer_name, dataset, epsilon, amount, session_id_val = default_inputs
    print(n_clicks)
    session_id = session_id_val[0]
    if n_clicks == 0:
        return "Please enter a value and click the submit button.", None, dataset, None
    elif dataset is None or dataset == "":
        return "Input value is required!", None, None, None

    print(f'Using synthesizer: {synthesizer_name}')
    synthesizer: Synthesizer = get_synthesizer_for_name(synthesizer_name)(f'../datasets/{dataset}.csv', session_id)
    result = synthesizer.request(epsilon=epsilon, num_tuples=amount, dataset=dataset, **sdv_options)
    df = pd.read_csv(f'temp/{session_id}.csv')
    MetaInformation(id=session_id, dataset_name=dataset).save()

    if isinstance(result, tuple):  # sdv synthesizer returns a tuple
        col_shapes_plt, col_pair_trends_plt = result
        col_shapes_plt.show()
        col_pair_trends_plt.show()
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", [], col_shapes_plt, col_pair_trends_plt, df.to_dict(
            'records'), [{'name': col, 'id': col} for col in df.columns]

    return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", result, [], [], df.to_dict('records'), [
        {'name': col, 'id': col} for col in df.columns]


def get_synthesizer_for_name(name: str):
    synthesizers = {
        'sdv': sdv_tvae_syn.SDVSynthesizer,
        'cds': cor_data_syn.CorrelatedDataSynthesizer,
        'ids': ind_data_syn.IndependentDataSynthesizer,
    }
    return synthesizers[name]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')
    app.run_server(port=8050)
