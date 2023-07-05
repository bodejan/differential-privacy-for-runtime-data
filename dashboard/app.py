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
from training import training_callbacks, training_layout


app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
)
app.title = "Synthetic Runtime Data"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Data Creation", href="/")),
        dbc.NavItem(dbc.NavLink("Runtime Prediction", href="/page1")),
        dbc.NavItem(dbc.NavLink("", href="/page2")),
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


homelayout = html.Div(
    [
        navbar,
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

    ])


app.layout = html.Div(
    children=[dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

training_callbacks(app)

@app.callback(
    [dash.dependencies.Output('csv-table', 'data'), 
    dash.dependencies.Output('csv-table', 'columns')],
    [dash.dependencies.Input('dropdown2', 'value')],
    [dash.dependencies.State('dropdown2', 'value')],
    prevent_initial_call=True)
def update_output(value, v2):
    df = pd.read_csv(f'../datasets{value}.csv')
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
        ds = cor_data_syn.CDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets{dataset}.csv', dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {value1} from menu 1 and {value2}, {value3}, {value4}, {dataset} from menu 2.", f'assets/temp_{dataset}.png', n_clicks, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
    else:
        ds = ind_data_syn.IDS(epsilon=value2, num_tuples=value4, input_data=f'../datasets{dataset}.csv', dataset=dataset)
        df = pd.read_csv('temp/sythetic_data.csv')
        return f"You selected {value1} from menu 1 and {value2}, {value3}, {value4}, {dataset} from menu 2.", f'assets/temp_ids_{dataset}.png', n_clicks, df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]



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
