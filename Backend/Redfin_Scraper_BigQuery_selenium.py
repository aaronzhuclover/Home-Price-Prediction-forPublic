import pandas as pd
import requests
import io
import time 
import random 
from datetime import datetime
import copy
import re
import numpy as np
import pickle

#######################################################################################
# connect using google.cloud package
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import tempfile
from pandas_gbq import schema

###################################################################
project = 'testpython-267102'

credentials = service_account.Credentials.from_service_account_file('google_creds.json')
client = bigquery.Client(credentials= credentials, project=project)

###################################################################
# scrape redfin data

region_id = pd.read_csv('Region_id.csv')
region_id = region_id[region_id.location.isin(['Rowland Heights', 'Arcadia', 'Walnut', 'Irvine', 'El Monte'])].reset_index(drop=True)

n = region_id.shape[0]

master = pd.DataFrame(columns = ['month', 'sale_volume', 'price_med', 'location'])
for i in range(n):
    # download all the historical sales data
    data = pd.DataFrame()
    # 2 Bedrooms or below
    repeat = 1
    while repeat:
        try: 
            print('We are working on ' + region_id['location'][i] + ' : bedroom ' + str(2))
            url = r'https://www.redfin.com/stingray/api/gis-csv?al=1&market=socal&max_num_beds=2&min_stories=1&num_homes=20000&ord=redfin-recommended-asc&page_number=1&region_id={}&region_type=6&sold_within_days=36500&status=9&uipt=1,2,3,4,5,6&v=8'.format(region_id['region_id'][i])
            print(url)
            page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).content
            time.sleep(1+random.random())
            s = pd.read_csv(io.StringIO(page.decode('utf-8')))
            repeat = 0
        except:
            repeat = 1

    data = data.append(s, ignore_index=True)
    # 3/4/5/6 Bedrooms   
    for x in range(3,7):
        repeat = 1
        while repeat:
            try: 
                try:
                    print('We are working on ' + region_id['location'][i] + ' : bedroom ' + str(x))
                    url = r'https://www.redfin.com/stingray/api/gis-csv?al=1&market=socal&max_num_beds={}&min_stories=1&num_beds={}&num_homes=20000&ord=redfin-recommended-asc&page_number=1&region_id={}&region_type=6&sold_within_days=36500&status=9&uipt=1,2,3,4,5,6&v=8'.format(x, x, region_id['region_id'][i])
                    print(url)
                    page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).content
                    time.sleep(1+random.random())
                    s = pd.read_csv(io.StringIO(page.decode('utf-8')))
                    repeat = 0
                    
                except: 
                    print('We are working on ' + region_id['location'][i] + ' : bedroom ' + str(x))
                    url = r'https://www.redfin.com/stingray/api/gis-csv?al=1&market=socal&max_num_beds={}&min_stories=1&num_beds={}&num_homes=20000&ord=redfin-recommended-asc&page_number=1&region_id={}&region_type=6&sold_within_days=36500&status=9&uipt=1,2,3,4,5,6&v=8'.format(x, x, region_id['region_id'][i])
                    print(url)
                    s = pd.read_csv(url)
                    time.sleep(1+random.random())
                    repeat = 0
            except:
                repeat = 1
                           
        data = data.append(s, ignore_index=True)  
        
    data.to_csv(r'\rawdata\{}.csv'.format(region_id['location'][i]), index = False)
    # need to clean data
    # by month/property type: (1) number of house sold (2) median price/sqft
    data = data[data['CITY']==region_id['location'][i]]
    data = data[data['PROPERTY TYPE']=='Single Family Residential']
    
    data = data[data['SOLD DATE'].isnull()==False]
    
    data['SOLD DATE2'] = pd.to_datetime(data['SOLD DATE'], format='%B-%d-%Y')
    data['year'] = pd.DatetimeIndex(data['SOLD DATE2']).year
    data['month'] = pd.DatetimeIndex(data['SOLD DATE2']).month
    data['SOLD DATE3'] = data['year'].astype(str) + '-' + data['month'].astype(str) + '-' + '01'
    data['SOLD DATE3'] = pd.to_datetime(data['SOLD DATE3'], format='%Y-%m-%d').dt.date
    
    # data['SOLD DATE3'] = data['SOLD DATE2'] + pd.offsets.MonthBegin(-1) # this is a trick of shortcout to change date
    table = data[['SOLD DATE3', '$/SQUARE FEET']].groupby(['SOLD DATE3']).agg(['count', 'median']).reset_index()    
    table.columns = ['month', 'sale_volume', 'price_med']
    table['location'] = region_id['location'][i]
    master = master.append(table)

print('*********************')
print('Finish downloading Redfin data!')

master['month'] = master['month'].astype(str)
master['sale_volume'] = master['sale_volume'].astype(int)

###################################################################
sql = '''
    DROP TABLE IF EXISTS `sampledata.df`;
    CREATE TABLE `sampledata.df`(
        month string, 
        sale_volume INT64,
    	price_med FLOAT64,
        location string
    );
    SELECT * FROM `sampledata.df`;
'''
query = client.query(sql)
results = query.result().to_dataframe()
print(results)


# insert using pd.to_gbq
# print(schema.generate_bq_schema(master))
schema_ = [
    {'name': 'month', 'type': 'STRING'},
    {'name': 'sale_volume', 'type': 'INTEGER'},
    {'name': 'price_med', 'type': 'FLOAT'},
    {'name': 'location', 'type': 'STRING'}
]

master.to_gbq(destination_table = 'sampledata.df',
          project_id = project,
          if_exists = 'append',
          credentials = credentials,
          table_schema=schema_
          )

###################################################################
# check sql
sql = '''
    SELECT * FROM `sampledata.df`;
'''
query = client.query(sql)
results = query.result().to_dataframe()
print(results)

###################################################################
print('*********************')
print('Clean last 7 days data')
# construct a SQL database that includes sales data from last 7 days. 

# import sales data from redfin
data = pd.DataFrame()
for i in ['Arcadia', 'El Monte', 'Irvine', 'Rowland Heights', 'Walnut']:
    one = pd.read_csv(r'\rawdata\{}.csv'.format(i))
    one = one[one['CITY'] == i]
    data = data.append(one)
    
data.rename(columns = {'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)':'URL'}, inplace = True)
data = data[~data['CITY'].isnull()]
data = data[data['CITY'].isin(['Irvine', 'Arcadia', 'El Monte', 'Walnut', 'Rowland Heights'])]
data = data[data['PROPERTY TYPE'].str.contains('Single|Condo|Townhouse')]
data = data[~data['SOLD DATE'].isnull()]
data.rename(columns = {'ZIP OR POSTAL CODE': 'ZIP', 
                          'HOA/MONTH': 'HOA',
                          'PROPERTY TYPE': 'PROPERTY_TYPE',
                          'SQUARE FEET': 'SQUARE_FEET',
                          'LOT SIZE': 'LOT_SIZE'
                          }, inplace = True)

data = data[~data['PRICE'].isnull()]
data['PRICE'] = data['PRICE'].astype('int')

# SOLD DATE: month 
data['SOLD DATE2'] = data.apply(lambda x: datetime.strptime(x['SOLD DATE'], '%B-%d-%Y').date() , axis = 1)

# keep sales data from last 7 days
data = data[data.apply(lambda x:  (datetime.now().date() - x['SOLD DATE2']).days <= 7  , axis = 1)]

data = data.drop(columns = ['SALE TYPE', 'SOLD DATE', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'FAVORITE', 'INTERESTED', 'STATUS', 'LOCATION', 'SOURCE', 'MLS#'])

data.columns = ['property_type', 'address', 'city', 'state', 'zip', 'price', 'beds', 'baths', 'sq_feet', 'lot_size', 'year_built', 'days_on_market', 'dollar_per_sq_feet', 'hoa', 'url', 'lat', 'long', 'sold_date']

# prediction
query = copy.deepcopy(data)
query = query.rename(columns = {'sq_feet':'square_feet'})
query['lot_size'] = query.apply(lambda x: x['lot_size'] if re.findall('Single', x['property_type']) else  x['square_feet'], axis =1)  
query['lot_size'] = query.apply(lambda x: x['square_feet'] if x['lot_size'] > 1000* x['square_feet'] else x['lot_size']  , axis =1) 

query['age'] = 2021 - query['year_built']
query['hoa'] = query.apply(lambda x: 0 if np.isnan(x['hoa']) else x['hoa'], axis =1)
query['mort_rate'] = 2.97
query['hpi'] = 323
query['property_type_Condo/Co-op'] = query.apply(lambda x: 1 if x['property_type'] == 'Condo/Co-op' else 0 , axis =1) 
query['property_type_Single Family Residential'] = query.apply(lambda x: 1 if x['property_type'] == 'Single Family Residential' else 0 , axis =1) 
query['property_type_Townhouse'] = query.apply(lambda x: 1 if x['property_type'] == 'Townhouse' else 0 , axis =1) 
for i in [91006, 91007, 91731, 91732, 91733, 91748, 91789, 92602, 92603, 92604, 92606, 92612, 92614, 92618, 92620]:
    query[f'zip_{i}'] = query.apply(lambda x: 1 if x['zip'] == i else 0 , axis =1) 
for i in range(1,13):
    query[f'mth_{i}'] = query.apply(lambda x: 1 if x['sold_date'].month == i else 0 , axis =1) 
for i in query['city'].unique():
    query[f'city_{i}'] = query.apply(lambda x: 1 if x['city'] == i else 0 , axis =1) 
 
    
query = query[['beds', 'baths', 'square_feet', 'lot_size', 'age', 'hoa', 'mort_rate',
       'hpi', 'property_type_Condo/Co-op',
       'property_type_Single Family Residential', 'property_type_Townhouse',
       'zip_91006', 'zip_91007', 'zip_91731', 'zip_91732', 'zip_91733',
       'zip_91748', 'zip_91789', 'zip_92602', 'zip_92603', 'zip_92604',
       'zip_92606', 'zip_92612', 'zip_92614', 'zip_92618', 'zip_92620',
       'mth_1', 'mth_2', 'mth_3', 'mth_4', 'mth_5', 'mth_6', 'mth_7', 'mth_8',
       'mth_9', 'mth_10', 'mth_11', 'mth_12', 'city_Arcadia', 'city_El Monte',
       'city_Irvine', 'city_Rowland Heights', 'city_Walnut']]

# load the ml model
xgb_model_deploy = pickle.load(open(r'\app\xgb_model_deploy.pickle', 'rb'))

data['pred_price'] = xgb_model_deploy.predict(query)
data['pred_price_diff'] = round(((data['price'] / data['pred_price'])-1)*100, 0)


###################################################################
#  insert into sql database
sql = '''
    DROP TABLE IF EXISTS `sampledata.sales_data`;
    CREATE TABLE `sampledata.sales_data`(
        property_type string,
        address string,
        city string,
        state string,
        zip FLOAT64,
        price INT64,
        beds FLOAT64,
        baths FLOAT64,
        sq_feet FLOAT64,
        lot_size FLOAT64,
        year_built FLOAT64,
        days_on_market FLOAT64,
        dollar_per_sq_feet FLOAT64,
        hoa FLOAT64,
        url string,
        lat FLOAT64,
        long FLOAT64,
        sold_date string,
        pred_price FLOAT64,
        pred_price_diff FLOAT64
    );
    SELECT * FROM `sampledata.sales_data`;
'''
query = client.query(sql)
results = query.result().to_dataframe()
print(results)

# insert using pd.to_gbq
# print(schema.generate_bq_schema(master))
schema_ = schema.generate_bq_schema(data)['fields']

data.to_gbq(destination_table = 'sampledata.sales_data',
          project_id = project,
          if_exists = 'append',
          credentials = credentials,
          table_schema=schema_
          )

##################
# check sql
sql = '''
    SELECT * FROM `sampledata.sales_data`;
'''
query = client.query(sql)
results = query.result().to_dataframe()
print(results)











    
