# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:59:07 2021

@author: aaron
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import tempfile
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import re 
from datetime import datetime, timedelta
import pickle
import numpy as np


###############################
print('****************************************************')
print('Current time: Connect to BigQuery : ' + str(datetime.now()))

# extract data from bigguery
project = 'testpython-267102'
credentials = service_account.Credentials.from_service_account_file('google_creds.json')
client = bigquery.Client(credentials= credentials, project=project)
###############################
# extract table 
sql = '''
    SELECT * FROM `sampledata.df`;
'''
query = client.query(sql)
df = query.result().to_dataframe()
df = df[['location', 'month', 'sale_volume','price_med']]
# df = df[df['location'] == "Rowland Heights"]
df = df.sort_values(by=['location', 'month'], ignore_index=True)

weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
sum_weights = np.sum(weights)
# compute weighted Moving average [need to move this part to scraping program]
df['price_med_ma6'] = df.groupby(['location'])['price_med'].rolling(6).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False).reset_index(drop = True).round(0)
df['year'] = df.month.str.slice(0,4).astype(int)
df['price_1yr_b4'] = df.groupby('location')['price_med_ma6'].shift(12)
df['yty_per_diff'] = df.apply(lambda x: x['price_med_ma6']/ x['price_1yr_b4'] - 1 , axis =1)

all_locations = df.location.unique()

###############################
# extract modified time of table 
sql = '''
    SELECT
      TIMESTAMP_MILLIS(last_modified_time) last_modified_time2
    FROM
      `testpython-267102.sampledata`.__TABLES__
    WHERE
      table_id = 'df';

'''
query = client.query(sql)
last_mofified_time = query.result().to_dataframe()
t = last_mofified_time.iloc[0,0] - timedelta(hours=7)  
last_mofified_time = re.sub('\..*', '', str(t))
###############################
# extract sales data 
# extract table 
sql = '''
    SELECT * FROM `sampledata.sales_data`;
'''
query = client.query(sql)
sales = query.result().to_dataframe()

###############################
print('****************************************************')
print('Current time: Create graphs : ' + str(datetime.now()))

# start the App
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets,
    # to verify your own website with Google Search Console
    meta_tags=[{'name': 'google-site-verification', 'content': 'Wu9uTwrweStHxNoL-YC1uBmrsXYFNjRCqmSQ8nNnNMs'}])
app.title = 'Housing Price Dashboard'
server = app.server

# load the ml model
xgb_model_deploy = pickle.load(open('xgb_model_deploy.pickle', 'rb'))

# import mapbox token
px.set_mapbox_access_token(open('mapbox.mapbox_token').read())

###############################


prediction_col1 =  dbc.Col([ 
                html.Br(),
                dbc.Row([html.H3(children='Predict Housing Price')]),
                dbc.Row([
                    dbc.Col(html.Label(children='City and Zip:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dcc.Dropdown(
                        id='city_zip',
                        options=[
                            {'label': 'Arcadia - 91006', 'value': 'Arcadia - 91006'},
                            {'label': 'Arcadia - 91007', 'value': 'Arcadia - 91007'},
                            {'label': 'El Monte - 91731', 'value': 'El Monte - 91731'},
                            {'label': 'El Monte - 91732', 'value': 'El Monte - 91732'},
                            {'label': 'El Monte - 91733', 'value': 'El Monte - 91733'},
                            {'label': 'Irvine - 92602', 'value': 'Irvine - 92602'},
                            {'label': 'Irvine - 92603', 'value': 'Irvine - 92603'},
                            {'label': 'Irvine - 92604', 'value': 'Irvine - 92604'},
                            {'label': 'Irvine - 92606', 'value': 'Irvine - 92606'},
                            {'label': 'Irvine - 92612', 'value': 'Irvine - 92612'},
                            {'label': 'Irvine - 92614', 'value': 'Irvine - 92614'},
                            {'label': 'Irvine - 92618', 'value': 'Irvine - 92618'},
                            {'label': 'Irvine - 92620', 'value': 'Irvine - 92620'},
                            {'label': 'Rowland Heights - 91748', 'value': 'Rowland Heights - 91748'},
                            {'label': 'Walnut - 91789', 'value': 'Walnut - 91789'},                            
                        ],
                        value='Walnut - 91789',
                        style = {"width": "50%", 'padding': '5px 0px 5px 10px', 'display': 'inline-block'}
                    )
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Beds:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dcc.Dropdown(
                        id='beds',
                        options=[
                            {'label': '1', 'value': 1},
                            {'label': '2', 'value': 2},
                            {'label': '3', 'value': 3},
                            {'label': '4', 'value': 4},        
                            {'label': '5', 'value': 5},
                            {'label': '6', 'value': 6},                 
                        ],
                        style = {"width": "30%", 'padding': '5px 0px 5px 10px' , 'display': 'inline-block'},
                        value=4
                    )
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Baths:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dcc.Dropdown(
                        id='baths',
                        options=[
                            {'label': '1', 'value': 1},
                            {'label': '1.5', 'value': 1.5},
                            {'label': '2', 'value': 2},
                            {'label': '2.5', 'value': 2.5},
                            {'label': '3', 'value': 3},   
                            {'label': '3.5', 'value': 3.5},         
                            {'label': '4', 'value': 4},
                            {'label': '5', 'value': 5},
                            {'label': '6', 'value': 6},                 
                        ],
                        style = {"width": "30%", 'padding': '5px 0px 5px 10px', 'display': 'inline-block' },
                        value=2.5
                    )
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Property Type:'), width={"order": "first"}, style = {'padding': '5px 0px 5px 0px'}),
                    dbc.RadioItems(
                        id='prop_type',
                        options=[
                            {'label': 'Single Family Residential', 'value': 'Single Family Residential'},
                            {'label': 'Condo/Co-op', 'value': 'Condo/Co-op'},
                            {'label': 'Townhouse', 'value': 'Townhouse'}
                            ],
                        style = {"width": "60%", 'padding': '5px 0px 10px 10px', 'display': 'inline-block' },
                        value = 'Single Family Residential',
                        labelStyle={'display': 'inline-block'},
                        inline=True # arrange list horizontally
                    )
                ]), 
                dbc.Row([
                    dbc.Col(html.Label(children='Square Feet:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='sf', type='text', value = '1980', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Lot Size:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='ls', type='text', value = '6000', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),  
                dbc.Row([
                    dbc.Col(html.Label(children='Year:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='year', type='text', value = '1977', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='HOA:'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='hoa', type='text', value = '0', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),                       
                html.Br(),
                dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
                html.Br(),
                dbc.Row([html.Div(id='container-button-basic')])
            ], style = {'padding': '0px 0px 0px 150px'})

prediction_col2 =  dbc.Col([ html.Br(), html.Div(dcc.Graph(id='hist-graph'))], style = {'padding': '0px 0px 0px 0px'})


sales_col_map = dbc.Col([
        html.Br(),
        dbc.RadioItems(
            id = 'recent_sales_city',
            options=[{"label": x, "value": x} for x in all_locations],
            value=all_locations[3],
            labelStyle={'display': 'inline-block'},
            labelCheckedStyle={"color": "red"},
            inline=True, # arrange list horizontally
            style={"justify-content":"space-between", "font-size":"24px", "margin-left": "0px"}
        ),
        html.Br(),
        html.Div(dcc.Graph(id='sales_map'))
    ], style = {'padding': '0px 0px 0px 60px'})

sales_col_table = dbc.Col([ 
        html.Br(),
        #html.Div(dash_table.DataTable(id='sales_table', columns=[{"name": i, "id": i} for i in sales.columns], data = sales.to_dict('records') )
        html.Div(children='Home Sales Within Last 7 Days'),
        html.Br(), 
        html.Div(id='sales_table')

    ], style = {'padding': '0px 0px 0px 20px'})

   

##############################################################
# prepare the layout
app.layout = html.Div([
    html.H1(children='Housing Price Index Dashboard'),

    html.Div(children='''Housing price is constructed using sales data from Redfin.'''),
    html.Div(children=f'''Last Updated: {last_mofified_time}'''),
    html.Div(children='Developer: Aaron Zhu'),
    html.Br(),

    dcc.Tabs(style = {'width': '100%'}, children=[
        # this is the first tab
        dcc.Tab(label='Housing Price Index', children = [
            html.Div([
                html.Br(),
                dbc.Row([html.Div(children='Choose Year Range', style = {"margin-left": "30px"})]),
                dbc.Row([
                    dbc.Col(
                        dcc.RangeSlider(
                            id='year-range-slider',
                            min=2000,
                            max=2021,
                            step=1,
                            value=[2020, 2021],
                            marks = {i: str(i) for i in range(2000,2022, 1)}
                        )
                    )
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dbc.RadioItems(
                            id="checklist",
                            options=[{"label": x, "value": x} for x in all_locations],
                            value=all_locations[3],
                            labelStyle={'display': 'inline-block'},
                            labelCheckedStyle={"color": "red"},
                            inline=True, # arrange list horizontally
                            style={"justify-content":"space-between", "font-size":"24px", "margin-left": "100px"}
                        )
                    )]),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id='line-graph'), style = {'width': '100%'}))
                ])
            ])
        ]), # the end of the second tab
        # this is the first tab
        dcc.Tab(label='Sales Volume', children = [
            html.Div([
                html.Br(),
                dbc.Row([html.Div(children='Choose Year Range', style = {"margin-left": "30px"})]),
                dbc.Row([
                    dbc.Col(
                        dcc.RangeSlider(
                            id='year-range-slider2',
                            min=2000,
                            max=2021,
                            step=1,
                            value=[2020, 2021],
                            marks = {i: str(i) for i in range(2000,2022, 1)}
                        )
                    )
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dbc.RadioItems(
                            id = 'sale_volume_city',
                            options=[{"label": x, "value": x} for x in all_locations],
                            value=all_locations[3],
                            labelStyle={'display': 'inline-block'},
                            labelCheckedStyle={"color": "red"},
                            inline=True, # arrange list horizontally
                            style={"justify-content":"space-between", "font-size":"24px", "margin-left": "70px"}
                        )
                    )
                ]),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id='bar-graph'), style = {'width': '100%'}))
                ])
            ])
        ]), # the end of the second tab

        # this is the 3rd tab
        dcc.Tab(label='Housing Price Prediction', children = [
            dbc.Row([prediction_col1, prediction_col2])
        ]), # the end of the 3rd tab

        # this is the 4th tab
        dcc.Tab(label='Recent Sales', children = [
            dbc.Row([sales_col_map]),
            dbc.Row([sales_col_table])
        ]) # the end of the 4th tab




    ]) # end of all tabs

], style = {'padding': '20px'}) # the end of app.layout


####################################################################################################################
# create callback for line graph
@app.callback(
    Output('line-graph', 'figure'),
    Input('checklist', 'value'),
    Input('year-range-slider', 'value')
)
def update_line_graph (city, year_range):
    selected_df = df[df.location==city]
    selected_df = selected_df.query(f'year>={year_range[0]} & year<={year_range[1]}') 
    
    trace1 = go.Bar(x=selected_df['month'], y =selected_df['yty_per_diff'], name = 'YOY Price Change', yaxis = 'y1', hovertemplate = '%{y:.0%}')
    trace2 = go.Scatter(x=selected_df['month'], y =selected_df['price_med_ma6'], name = 'Price ($/SF)', mode='lines+markers', yaxis = 'y2', hovertemplate = '$%{y}')
    data = [trace2, trace1]
    fig = go.Figure(data=data)
    # update figure layout
    fig.update_layout(
        title  = {'text': f'Median Single House Price in {city}', 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 20}},
        legend = {'orientation': 'h', 'yanchor': "bottom", 'xanchor': "left", 'x': 0, 'y': 1, 'font': {'size': 15} },
        legend_title='',
        hovermode="x unified",
        #hover_data={'yty_per_diff': ':.0%'},
        hoverlabel = {'font_size': 12, 'font_family': "Rockwell", 'namelength': 50},
        yaxis=dict(title = 'YOY Price Change', side = 'right', tickformat = '%', dtick=0.05, showgrid = False),
        yaxis2 = dict(title = 'Price ($/SF)', overlaying = 'y', side = 'left')
    )
    fig.add_annotation(x = 0, y = -0.2, text = 'Notes: A 6-month weighted moving average is used to smooth the price series.', 
        showarrow = False, xref='paper', yref='paper', font = {'size': 15})
     
    return fig

#############################
# create callback for bar graph
@app.callback(
    Output('bar-graph', 'figure'),
    Input('sale_volume_city', 'value'),
    Input('year-range-slider2', 'value')
)
def update_bar_graph (city, year_range):
    selected_df = df[df.location==city]
    selected_df = selected_df.query(f'year>={year_range[0]} & year<={year_range[1]}') 
    fig = px.bar(selected_df, x="month", y="sale_volume", title=f'Sale Volume in {city}',
        labels = {'sale_volume':'Sale Volumn', 'month': ''}, height=500
    )
    fig.update_layout(
        title  = {'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 20}},
        hovermode = 'x unified',
        hoverlabel = {'font_size': 16, 'font_family': "Rockwell", 'namelength': 20,'font': {'size': 20}}
    )
    fig.update_traces(hovertemplate=None)
    
    return fig
##############################
# create call back fror prediction
@app.callback(
    Output('container-button-basic', 'children'),
    Output('hist-graph', 'figure'),
    # Inputs will trigger your callback; State do not. If you need the the current “value” - aka State - of other dash components within your callback, you pass them along via State.
    Input('submit-val', 'n_clicks'),
    State('city_zip', 'value'),
    State('beds', 'value'),
    State('baths', 'value'), 
    State('prop_type', 'value'),
    State('sf', 'value'),
    State('ls', 'value'),
    State('year', 'value'),
    State('hoa', 'value')) 
   
def update_output(n_clicks, city_zip, beds, baths, prop_type, sf, ls, year, hoa):
    if int(hoa) == 0:
        hoa = 0.01
    else:
        hoa = float(hoa)

    city_string = city_zip.split(' - ')[0]
    zip_string = city_zip.split(' - ')[1]


    query = pd.DataFrame({'beds': beds,
                        'baths':baths,
                        'square_feet':float(sf),
                        'lot_size':float(ls),
                        'age':2021-int(year),                      
                        'hoa':hoa,
                        'mort_rate':2.92,           
                        'hpi': 334,
                        'property_type_Condo/Co-op':0,
                        'property_type_Single Family Residential':0,
                        'property_type_Townhouse':0,                      
                        'zip_91006':0,
                        'zip_91007':0, 
                        'zip_91731':0,
                        'zip_91732':0,
                        'zip_91733':0, 
                        'zip_91748':0,
                        'zip_91789':0,
                        'zip_92602':0, 
                        'zip_92603':0,
                        'zip_92604':0,
                        'zip_92606':0, 
                        'zip_92612':0,
                        'zip_92614':0,
                        'zip_92618':0, 
                        'zip_92620':0,                      
                        'mth_1':0,
                        'mth_2':0,  
                        'mth_3':0,
                        'mth_4':0,  
                        'mth_5':1,
                        'mth_6':0,  
                        'mth_7':0,
                        'mth_8':0,  
                        'mth_9':0,
                        'mth_10':0,  
                        'mth_11':0,
                        'mth_12':0,
                        'city_Arcadia':0,
                        'city_El Monte':0,
                        'city_Irvine':0,
                        'city_Rowland Heights':0,
                        'city_Walnut':0
                        }, index = [0])

    query[f'city_{city_string}'] = 1
    query[f'zip_{zip_string}'] = 1
    query[f'property_type_{prop_type}'] = 1

    prediction = int(xgb_model_deploy.predict(query)[0].round(-3))
    output = int(prediction)

    pos = prediction
    scale = xgb_model_deploy.st_dev
    size = 200
    np.random.seed(123)
    values = np.random.normal(pos, scale, size)
    his_df = pd.DataFrame(values, columns = ['Price'])

    p_25 = int((pos-0.67*scale).round(-3))
    p_50 = pos
    p_75 = int((pos+0.67*scale).round(-3))
    dollar_per_sq_ft = int(p_50/query['square_feet'])

    fig = px.histogram(his_df, x = 'Price', histnorm ='percent', nbins = 20, width = 00, height  = 600)
    fig.add_vline(x=p_25, line_width=3, line_dash="dash", line_color="green", annotation_text=f"25th Percentile: ${p_25:,}", annotation_position="top left")
    fig.add_vline(x=p_75, line_width=3, line_dash="dash", line_color="green", annotation_text=f"75th Percentile: ${p_75:,}", annotation_position="top right")
    fig.add_vline(x=p_50, line_width=3, line_dash="dash", line_color="green", annotation_text=f"50th Percentile (Median Price): ${p_50:,}", annotation_position="top")
    fig.add_vrect(x0=p_25, x1=p_75, line_width=0, fillcolor="red", opacity=0.2)


    return f'The estimated median price is ${p_50:,} (${dollar_per_sq_ft} psf). The 50% CI is [${p_25:,}, ${p_75:,}].' , fig
##############################
# create call back for recently sales
@app.callback(
    Output('sales_table', 'children'),
    Output('sales_map', 'figure'),
    Input('recent_sales_city', 'value')
)
def update_recent_sales_table (city):
    selected_df = sales[sales.city==city]
    selected_df = selected_df[['url', 'pred_price_diff', 'property_type', 'sold_date', 'address', 'price', 'dollar_per_sq_feet', 'hoa', 'beds', 'baths', 'sq_feet', 'lot_size', 'year_built', 'days_on_market', 'lat', 'long']]
    selected_df = selected_df.to_dict('records')
    for i in selected_df:
        i['url'] = [html.A(html.P('Redfin Link'), href=i['url'], target="_blank")]
    selected_df = pd.DataFrame(selected_df)
    selected_df = selected_df.sort_values(['dollar_per_sq_feet'], ascending = [True])
    selected_df['price'] = selected_df.apply(lambda x:  f'${x["price"]:,}', axis =1)
    selected_df['dollar_per_sq_feet'] = selected_df.apply(lambda x:  f'${int(x["dollar_per_sq_feet"])}', axis =1)
    selected_df.columns = ['URL', 'Delta %', 'Property Type', 'Sold Date', 'Address', 'Price', '$ PSF', 'HOA','Beds', 'Baths', 'SQ Feet', 'Lot Size', 'Year Built', 'DOM', 'Lat', 'Long']
    sale_table = dbc.Table.from_dataframe(selected_df.drop(columns = ['Lat', 'Long']), striped=True, bordered=True, hover=True, size = 'sm')
     
    #################
    fig = px.scatter_mapbox(data_frame=selected_df, lat='Lat', lon='Long', 
            #opacity=0.5, 
            hover_name="Address", 
            hover_data=["Price"],
            zoom=10, width=600, height=400
        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},  # remove the white gutter between the frame and map
            # hover appearance
            hoverlabel=dict( 
                bgcolor="white",     # white background
                font_size=16,        # label font size
                font_family="Inter") # label font
        )



    return sale_table, fig

#############################
print('****************************************************')
print('Current time: Run App : ' + str(datetime.now()))
# run the app 
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    