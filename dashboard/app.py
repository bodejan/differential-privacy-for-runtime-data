"""
    A simple dash interface for creating synthetic data for runtime prediction

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""



import pandas as pd

import os
import uuid

import dash
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table



import cor_data_syn 
import ind_data_syn
import sdv_tvae_syn
from training_app import training_callbacks, training_layout
from regression_app import regression_layout
import eval_nn


app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
)
app.title = "Synthetic Runtime Data"

navbar = html.Div(
    html.H1('Synthetic Data for Runtime Prediction', style={'color': 'white', 'margin': '0'}),
    style={
        'background-color':'#0b1120',
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
                html.P("Select Epsilon"),
                dcc.Slider(0, 1, 0.1, value=0, id='epsilon'),
                html.Br(),
                html.P("Amount of Data to Generate"),
                dbc.Input(id = "amount", type="number", value = 1000, min=10, max=100000, step=10),
                html.Br(),
                html.Div([
                    dcc.Loading(id="loading",type="circle",
                                children=[
                                    html.Button('Create', id='create-button', n_clicks=0, style={'align': 'center', 'width':'100%', 'display': 'inline-block', 'background-color': '#4CAF50', 'color': 'white'}),
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

            ], style={'marginLeft':"0.5cm", 'height':'100%'}))
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
            ], style={'height':'100%'}))
    ], width=8),
])

app.layout= html.Div(
    [
        navbar,
        html.Div(session_id(), id='session-id'),# style={'display': 'none'}),
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
    [dash.dependencies.Output('output', 'children'),
    dash.dependencies.Output('eval_image', 'src'), 
    dash.dependencies.Output('csv-table-synthetic', 'data'),
    dash.dependencies.Output('csv-table-synthetic', 'columns')],
    [dash.dependencies.State('synthesizer', 'value'),
     dash.dependencies.State('dataset', 'value'),
     dash.dependencies.State('epsilon', 'value'),
     dash.dependencies.State('amount', 'value'),
    dash.dependencies.State('session-id', 'children')],
    [dash.dependencies.Input('create-button', 'n_clicks')],prevent_initial_call=True)

def update_output(synthesizer, dataset, epsilon, amount, session_id_val, n_clicks):
    print(n_clicks)
    if n_clicks == 0 :
        return "Please enter a value and click the submit button.",None, dataset, None
    elif dataset is None or dataset =="":
        return "Input value is required!", None, None, None
    elif synthesizer == "sdv":
        print("Called SDV synthesizer")
        sdv_tvae = sdv_tvae_syn.SDV_TVAE()
        col_shapes_plt, col_pair_trends_plt = sdv_tvae.request(f'../datasets/{dataset}.tsv', session_id_val[0],amount, 10, 10 )
        print("Finished sdv")
        df = pd.read_csv(f'temp/{session_id_val[0]}.csv')
        col_shapes_plt.show()
        col_pair_trends_plt.show()
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", col_shapes_plt, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    elif synthesizer == "cds":
        print("ID", session_id_val)
        ds = cor_data_syn.CDS(epsilon=epsilon, num_tuples=amount, input_data=f'../datasets/{dataset}.csv', uuid=session_id_val[0], dataset=dataset)
        df = pd.read_csv(f'dashboard/temp/{session_id_val[0]}.csv')
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", f'../dashboard/assets/temp_{dataset}.png', df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    else:
        print("ID", session_id_val)
        ids = ind_data_syn.IDS()
        figure = ids.request(epsilon=epsilon, num_tuples=amount, input_data=f'../datasets/{dataset}.csv', uuid=session_id_val[0], dataset=dataset, )
        df = pd.read_csv(f'temp/{session_id_val[0]}.csv')
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", figure, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    app.run_server(port=8050)
