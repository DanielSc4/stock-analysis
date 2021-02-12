import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime as dt
import dash_table

import numpy as np
import pandas as pd
import pandas_datareader.data as web

def download_data(stocks = ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"], start_stream = '2006-01-01'):
    # dictionary witch contains the name and the dataframe
    dataframes = {}
    end_stream = dt.now()
    # adds stock's dataframe in dataframes dictionary
    for s in stocks:
        print("[] downloading " + s + "\t -> ", end = "")
        dataframes[s] = web.get_data_yahoo(s, start_stream, end_stream)
        print("Done")
    
    return dataframes





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

g_budget = '   click refresh to update the value'
g_date = '   click refresh to update the value'
g_expected_return_volat = '   click refresh to update the value'
g_recap = pd.DataFrame()
g_spent = '   click refresh to update the value'
g_left = '   click refresh to update the value'
g_returns = '   click refresh to update the value'

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([
    html.H1('Markowitz Portfolio Calculator'),
    html.Br(),
    html.Br(),
    html.Strong(html.Label('Select your budget')),
    dcc.Slider(
        id='slider',
        marks={i: str(i) for i in [100,10000,20000,30000]},
        min=100,
        max=30000,
        value=[15000]),
    html.Br(),
    html.Br(),
    html.Strong(html.Label('Date Range Investment')),
    dcc.DatePickerRange(id='date-picker-range',
                        start_date=dt(2019,12,31),
                        initial_visible_month=dt.today(),
                        end_date_placeholder_text='Select a date!'),
    html.Br(),
    html.Br(),

    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-budget'),
        html.Div(children='Budget selected'),
        html.Div(id='budget'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-date'),
        html.Div(children='Date range selected'),
        html.Div(id='date'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-ex_rt_v'),
        html.Div(children='Expected return and volatility'),
        html.Div(id='ex_rt_v'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-recap'),
        html.Div(children = 'Number of shares, price for each share, transaction cost, purchase cost'),
        dcc.Graph(id = 'recap'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-spent'),
        html.Div(children='Total spent on the investment'),
        html.Div(id='spent'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-left'),
        html.Div(children='Total left from the investment'),
        html.Div(id='left'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Button('Refresh values', id='check-returns'),
        html.Div(children='Real investment return'),
        html.Div(id='returns'),
        html.Br()
    ]),
    html.Div(children = '_____________________________'),
    html.Div([
        html.Br(),
        html.Div(children='end'),
        html.Div(id='end'),
        html.Br()
    ])

   ])



@app.callback(Output(component_id='budget', component_property='children'), 
                [Input(component_id='check-budget', component_property='n_clicks')])
def update_budget(_):
    print(g_expected_return_volat)
    return str(g_budget)

@app.callback(Output(component_id='date', component_property='children'), 
                [Input(component_id='check-date', component_property='n_clicks')])
def update_date(_):
    return str(g_date)

@app.callback(Output(component_id='ex_rt_v', component_property='children'), 
                [Input(component_id='check-ex_rt_v', component_property='n_clicks')])
def update_exp_rt(_):
    return str(g_expected_return_volat)

@app.callback(Output(component_id='recap', component_property='figure'), 
                [Input(component_id='check-recap', component_property='n_clicks')])
def update_recap(_):
    # g_recap
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Table(
        header = dict(values = ['Number of shares', 'Price for each share', 'Transaction cost', 'Purchase cost'],
                    line_color = '#000000',
                    fill_color = '#5963f3',
                    align = 'center',
                    font = dict(color='white')),
        cells = dict(values=[list(g_recap[i]) for i in g_recap.columns], # 1st and 2nd and so ... columns
                line_color = '#000000',
                fill_color = '#9da6f3',
                align = 'center',
                font = dict(color='white')))
                ])
    # fig.show()

    return fig

@app.callback(Output(component_id='spent', component_property='children'), 
                [Input(component_id='check-spent', component_property='n_clicks')])
def update_spent(_):
    return str(g_spent)

@app.callback(Output(component_id='left', component_property='children'), 
                [Input(component_id='check-left', component_property='n_clicks')])
def update_left(_):
    return str(g_left)

@app.callback(Output(component_id='returns', component_property='children'), 
                [Input(component_id='check-returns', component_property='n_clicks')])
def update_returns(_):
    return str(g_returns)






stocks = ['AAPL', 'NVDA', 'KO', 'UL', 'BAC', 'AXP']

start_period = '2006-12-01'
# yesterday = '2018-12-31' # yesterday day before the creation of the portfolio 
# end_period = '2019-12-31' # end_period end of investment 


dataframes = download_data(stocks = stocks, start_stream = start_period)

weights = {}
for s in stocks:
    weights[s] = np.nan


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

 
@app.callback(Output('end', 'children'), 
              [Input('slider', 'value'), 
              Input(component_id='date-picker-range', component_property='start_date'),
              Input(component_id='date-picker-range', component_property='end_date')])
def mo_portfolio(budget, yesterday, end_period):

    # debug
    print(f'Hello from mo_portfolio')
    print(f'- budget:\t\t {budget}')
    print(f'- yesterday:\t {yesterday}')
    print(f'- end_period:\t {end_period}')

    if end_period is None:
        return 'False'

    global g_budget
    g_budget = budget
    global g_date
    g_date = [yesterday, end_period]
    # yesterday: day before the creation of the portfolio 
    # end_period: end of investment 

    chart = pd.DataFrame()
    # chart: adj closes fino a yesterday
    for s in stocks:
        chart = pd.concat([chart, dataframes[s]['Adj Close'].loc[:yesterday,]], axis = 1)
    chart.columns = stocks
    # compute montly (default value = 'Y') cc return 
    chart_rt = {}
    for s in chart:
        tmp = chart[s].groupby(pd.Grouper(freq = "Y"))
        tmp2 = tmp.mean()
        chart_rt[s] = np.log(tmp2/tmp2.shift(1))
    chart_rt = pd.DataFrame.from_dict(chart_rt)
    chart_rt = chart_rt.dropna()
    chart_rt.columns = ["AAPL CC returns", "NVDA CC returns", "KO CC returns", "UL CC returns", "BAC CC returns", "AXP CC returns"]

    # adding transition costs (1,5% fee per share)
    chart = chart.apply(lambda x: x + (x * 0.015))

    # Optimal portfolio

    # computes CC return on year granularity
    avg_returns = expected_returns.mean_historical_return(chart)
    # sample covariance matrix 
    S = risk_models.sample_cov(chart)
    ef = EfficientFrontier(avg_returns, S)

    # Minimize the volatily of the portfolio (Markowitz)
    weights = ef.min_volatility()
    # rounding weights values, meaning they may not add up exactly to 1 but should be close
    weights = ef.clean_weights()

    Mop_pw = weights

    opt_return, opt_risk, _ = ef.portfolio_performance(verbose=False)
    global g_expected_return_volat
    g_expected_return_volat = [opt_return, opt_risk]
    
    recap = {}
    for s in weights:
        # print(f'{s} budget {budget}, {type(budget)}')     # debug
        # print(f'{s} weights[s]/chart[s].iloc[-1] {weights[s]/chart[s].iloc[-1]}, {type(weights[s]/chart[s].iloc[-1])}')   # debug
        recap[s] = [int(np.floor(budget * weights[s]/chart[s].iloc[-1]))] # number of shares
        price_no_fee = np.round(chart[s].iloc[-1] - (chart[s].iloc[-1] * 1.5 / 101.5), decimals = 2)
        recap[s].append(price_no_fee) # price for each shares
        recap[s].append(np.round(price_no_fee * 0.015, 2)) # transaction costs 1,5%
        tot_cost = np.around(recap[s][0] * (recap[s][1] + recap[s][2]), decimals = 2)
        recap[s].append(tot_cost) # total cost of the investment in s (shares * (price for each s + transaction costs))

    recap = pd.DataFrame.from_dict(recap, orient='index')
    recap.columns = ['Num of shares', 'Price for each share $', 'Transaction costs $', 'Purchase cost $']

    global g_recap
    g_recap = recap

    total = 0
    for _, row in recap.iterrows():
        total += row['Purchase cost $']

    total = np.around(total, decimals = 2)

    global g_spent 
    g_spent = total
    global g_left 
    g_left = str(np.around(budget - total, decimals = 2))

    price_end = {}
    tot_port = 0
    for s in dataframes:
        price_end[s] = dataframes[s]['Adj Close'].loc[end_period]

    act_return = 0
    for index, row in recap.iterrows():
        tot_port += np.around(row['Num of shares'] * (price_end[index] + row['Transaction costs $']), decimals = 2)
        rtn = (price_end[index] + row['Transaction costs $'])/recap.loc[index,'Price for each share $'] - 1
        act_return += weights[index] * rtn

    global g_returns
    g_returns = str(np.around(tot_port, decimals = 2)) + ' [' + str(np.round(100*act_return, decimals = 2)) + '%]'
    print(g_returns)

    return "True"





if __name__ == '__main__':
    app.run_server()