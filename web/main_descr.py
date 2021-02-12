# hello

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas_datareader.data as web
from datetime import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px # smarter than matplotlib?


import dash_table
import dash_bootstrap_components as dbc # pip install dash-bootstrap-components


body = dbc.Container([
    html.Br(),
    dbc.Row([html.H1("Descriptive Analytics")], justify="center"),
    html.Br(),
    html.H3('Your stock selection:'),
    dcc.Dropdown(
        id='dropdown_stock',
        options=[
                {'label': 'Apple Inc.', 'value': 'AAPL'},
                {'label': 'Nvidia', 'value': 'NVDA'},
                {'label': 'The Coca Cola Company', 'value': 'KO'},
                {'label': 'Unilever', 'value': 'UL'},
                {'label': 'Bank Of America', 'value': 'BAC'},
                {'label': 'American Express', 'value': 'AXP'}
        ],
        value='AAPL'
    ),
    html.Br(),
    html.H3('Adj Close: '),
    dcc.Graph(id='plt_adjclose'),

    html.Br(),
    html.H3('CC returns:'),
    dcc.Graph(id='plt_returns'),

    html.Br(),
    html.H3('Histogram and boxplot:'),
    html.Div(children=[
        html.Br(),
        dcc.Graph(id = 'plt_hist'), # , style={'display': 'inline-block'}
        html.Br(),
        dcc.Graph(id = 'plt_boxplot') # , style={'display': 'inline-block'}
    ]),

    html.Br(),
    html.H3('Oth return Stats:'),
    dcc.Graph(id = 'table'),
    html.Br(),

    # https://plotly.com/python/table/#tables-in-dash
    #dash_table.DataTable(
    #    id='table',
    #    columns=[{"name": i, "id": i} 
    #             for i in df.columns],
    #    data=df.to_dict('records'),
    #    style_cell=dict(textAlign='left'),
    #    style_header=dict(backgroundColor="paleturquoise"),
    #    style_data=dict(backgroundColor="lavender")
    #)


    ],style={"height": "100vh"})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

app.layout = html.Div([body])

@app.callback(Output('table', 'figure'), [Input('dropdown_stock', 'value')])
def update_table(stock_name):
    df = web.DataReader(stock_name, data_source='yahoo', start=dt(2007, 1, 1), end=dt.now())
    df = df.groupby(pd.Grouper(freq='M')).mean()

    df_adj = df["Adj Close"]
    df_adj_op = df_adj.groupby(pd.Grouper(freq = 'M'))
    df_adj = df_adj_op.mean()
    df_returns = np.log(df_adj/df_adj.shift(1)) # calculating CC returns
    df_returns.name = stock_name + " CC Return"
    df_returns.dropna()

    desc = df_returns.describe()

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Table(
        header = dict(values = ['Stats', 'Value'],
                    line_color = '#000000',
                    fill_color = '#5963f3',
                    align = 'center',
                    font = dict(color='white')),
        cells = dict(values=[list(desc.index), list(desc)], # 1st and 2nd column
                line_color = '#000000',
                fill_color = '#9da6f3',
                align = 'center',
                font = dict(color='white')))
                ])
    return fig


@app.callback(Output('plt_adjclose', 'figure'), [Input('dropdown_stock', 'value')])
def update_graph_adj(stock_name):
    df = web.DataReader(stock_name, data_source='yahoo', start=dt(2007, 1, 1), end=dt.now())
    df = df.groupby(pd.Grouper(freq='M')).mean()
    fig = px.line(df['Adj Close'], y = 'Adj Close', title = 'Adj Cl. ' + stock_name)
    return fig

@app.callback(Output('plt_returns', 'figure'), [Input('dropdown_stock', 'value')])
def update_graph_returns(stock_name):
    df = web.DataReader(stock_name, data_source='yahoo', start=dt(2007, 1, 1), end=dt.now())
    df = df.groupby(pd.Grouper(freq='M')).mean()

    df_adj = df["Adj Close"]
    df_adj_op = df_adj.groupby(pd.Grouper(freq = 'M'))
    df_adj = df_adj_op.mean()
    df_returns = np.log(df_adj/df_adj.shift(1)) # calculating CC returns
    df_returns.name = stock_name + " CC Return"
    df_returns.dropna()

    fig = px.line(df_returns, title = stock_name + 'CC return')
    return fig

@app.callback(Output('plt_hist', 'figure'), [Input('dropdown_stock', 'value')])
def update_graph_hist(stock_name):
    df = web.DataReader(stock_name, data_source='yahoo', start=dt(2007, 1, 1), end=dt.now())
    df = df.groupby(pd.Grouper(freq='M')).mean()

    df_adj = df["Adj Close"]
    df_adj_op = df_adj.groupby(pd.Grouper(freq = 'M'))
    df_adj = df_adj_op.mean()
    df_returns = np.log(df_adj/df_adj.shift(1)) # calculating CC returns
    df_returns.name = stock_name + " CC Return"
    df_returns.dropna()

    fig = px.histogram(df_returns, title = stock_name + 'Histogram Plot')
    return fig

@app.callback(Output('plt_boxplot', 'figure'), [Input('dropdown_stock', 'value')])
def update_graph_boxplot(stock_name):
    df = web.DataReader(stock_name, data_source='yahoo', start=dt(2007, 1, 1), end=dt.now())
    df = df.groupby(pd.Grouper(freq='M')).mean()

    df_adj = df["Adj Close"]
    df_adj_op = df_adj.groupby(pd.Grouper(freq = 'M'))
    df_adj = df_adj_op.mean()
    df_returns = np.log(df_adj/df_adj.shift(1)) # calculating CC returns
    df_returns.name = stock_name + " CC Return"
    df_returns.dropna()

    fig = px.box(df_returns, title = stock_name + 'Boxplot')
    return fig




if __name__ == '__main__':
    app.run_server()