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
from training import training_callbacks, training_layout
import eval_nn


app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
)
app.title = "Synthetic Runtime Data"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Data Creation", href="/")),
        dbc.NavItem(dbc.NavLink("Runtime Prediction with neural Network", href="/page1")),
        dbc.NavItem(dbc.NavLink("Regression", href="/page2")),
    ],
    brand="Synthetic Data for Runtime Prediction",
    color="#119dff",
    brand_href="#",

    dark=True,
    style={
        "height": "80px",
        "line-height": "80px",
    },
)

page1_layout = dbc.Container(
    [navbar,training_layout()],
    fluid=True,
)
# Define the page 2 layout
page2_layout = dbc.Container(
    [navbar], #creator_layout()],
    fluid=True,
)

def session_id():
    return str(uuid.uuid4()),


homelayout = html.Div(
    [
        navbar,
        html.Div(session_id(), id='session-id'),#, style={'display': 'none'}),
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
                            html.Button("Download Text", id="btn-download-txt"),
                            dcc.Download(id="download-text")

                ], style={'margin':"0.5cm", 'height':'100%'}))
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
        ]),

    ])


app.layout = html.Div(
    children=[dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

training_callbacks(app)

@app.callback(
    Output("download-text", "data"),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(
        "synthetic_data.csv"
    )


@app.callback(
    [dash.dependencies.Output('csv-table-original', 'data'),
    dash.dependencies.Output('csv-table-original', 'columns')],
    [dash.dependencies.Input('dataset', 'value')],
    prevent_initial_call=True)
def update_output(value):
    df = pd.read_csv(f'datasets/{value}.csv')
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
    dash.dependencies.State('session-id', 'value')],
    [dash.dependencies.Input('create-button', 'n_clicks')],prevent_initial_call=True)

def update_output(synthesizer, dataset, epsilon, amount,session_id, n_clicks):
    print(n_clicks)
    if n_clicks == 0 :
        return "Please enter a value and click the submit button.", dataset, n_clicks
    elif dataset is None or dataset =="":
        return "Input value is required!", "", n_clicks, "", ""
    elif synthesizer == "cds":
        ds = cor_data_syn.CDS(epsilon=epsilon, num_tuples=amount, input_data=f'datasets/{dataset}.csv', uuid=session_id, dataset=dataset)
        df = pd.read_csv('dashboard/temp/sythetic_data.csv')
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", f'dashboard/assets/temp_{dataset}.png', df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    else:
        ids = ind_data_syn.IDS
        figure = ids.request(epsilon=epsilon, num_tuples=amount, input_data=f'../datasets/{dataset}.csv',  uuid=session_id, dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {synthesizer}, {dataset}, {epsilon}, {amount}.", figure, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]



    # Callback to update the page content based on the URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/page1":
        return page1_layout
    elif pathname == "/page2":
        return page2_layout
    else:
        return homelayout


if __name__ == "__main__":
    app.run_server(port=8050)
