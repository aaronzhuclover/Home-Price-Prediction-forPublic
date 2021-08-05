# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:45:10 2021

@author: aaron
"""

from fredapi import Fred
import pandas as pd
from datetime import datetime, date, time, timedelta
import numpy as np 
import re 
import pickle

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet

import matplotlib.pyplot as plt

################################################################
fred = Fred(api_key='3e45db934f364bc329aca420c85fa04e')
# extract S&P/Case-Shiller CA-Los Angeles Home Price Index (LXXRNSA)
# https://fred.stlouisfed.org/series/LXXRNSA
la_hpi_raw = fred.get_series('LXXRNSA')
la_hpi = la_hpi_raw.to_frame()
la_hpi.columns = ['hpi']
la_hpi['month'] = la_hpi.index
la_hpi = la_hpi.reset_index(drop = True)
la_hpi['month'] = la_hpi.apply(lambda x: x['month'].date(), axis = 1)

weights = np.array([0.2, 0.3, 0.5])
sum_weights = np.sum(weights)
# compute weighted MA from latest 3 months
la_hpi['hpi'] = la_hpi['hpi'].rolling(3).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False).reset_index(drop = True)
month_diff = datetime.now().month - la_hpi['month'].max().month
# reset the month variable so that it can be merged with sales data
la_hpi['month'] = la_hpi['month'] + pd.DateOffset(months=month_diff)
la_hpi['month'] = la_hpi.apply(lambda x: x['month'].date(), axis = 1)

################################################################
# 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US)
# https://fred.stlouisfed.org/series/MORTGAGE30US
mort_rate_30yrs_raw = fred.get_series('MORTGAGE30US')

mort_rate_30yrs = mort_rate_30yrs_raw.to_frame()
mort_rate_30yrs.columns = ['mort_rate']
mort_rate_30yrs['date'] = mort_rate_30yrs.index
mort_rate_30yrs = mort_rate_30yrs.reset_index(drop = True)
mort_rate_30yrs['date'] = mort_rate_30yrs.apply(lambda x: x['date'].date(), axis = 1)
mort_rate_30yrs['year'] = mort_rate_30yrs.apply(lambda x: x['date'].year, axis = 1)
mort_rate_30yrs['month'] = mort_rate_30yrs.apply(lambda x: x['date'].month, axis = 1)
mort_rate_30yrs = mort_rate_30yrs.groupby(['year', 'month'])['mort_rate'].mean().reset_index()
mort_rate_30yrs['month'] = mort_rate_30yrs.apply(lambda x: date(int(x['year']), int(x['month']), 1), axis = 1)
mort_rate_30yrs = mort_rate_30yrs.drop(columns = ['year'])

################################################################
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

##################################
# start cleaning redfine data
# PRICE: Y
data = data[~data['PRICE'].isnull()]
data['PRICE'] = data['PRICE'].astype('int')

# SOLD DATE: month 
data['SOLD DATE2'] = pd.to_datetime(data['SOLD DATE'], format='%B-%d-%Y')
data['year'] = pd.DatetimeIndex(data['SOLD DATE2']).year
data['mth'] = pd.DatetimeIndex(data['SOLD DATE2']).month
data['month'] = data.apply(lambda x: date(int(x['year']), int(x['mth']), 1), axis = 1)

# YEAR BUILT: age 
# age should be the age on the sold date
data['age'] = data['year'] - data['YEAR BUILT']

# HOA/MONTH: if nan, set it to be $0.01
data['HOA'] = data.apply(lambda x: 0.01 if (np.isnan(x['HOA']) & bool(re.findall('Single', x['PROPERTY_TYPE'])))  else  x['HOA'], axis =1)

# LOT SIZE: change LOT SIZE to be  SQUARE FEET for non-single house
data['LOT_SIZE'] = data.apply(lambda x: x['LOT_SIZE'] if re.findall('Single', x['PROPERTY_TYPE']) else  x['SQUARE_FEET']    , axis =1)

# ZIP
data = data[~data['ZIP'].isnull()]
data['ZIP'] = data['ZIP'].astype('int')
zip_keep = data['ZIP'].value_counts()
zip_keep = zip_keep[zip_keep/len(data) > 0.01].index
data = data[data['ZIP'].isin(zip_keep)]

data['ZIP'].value_counts(dropna = False)

keep_var = ['PRICE', 'PROPERTY_TYPE', 'CITY', 'ZIP', 'BEDS', 'BATHS', 'SQUARE_FEET', 'LOT_SIZE', 'age', 'HOA', 'year', 'mth', 'month']

data = data[keep_var]

df = data.merge(mort_rate_30yrs, on = 'month', how = 'left')
df = df.merge(la_hpi, on = 'month', how = 'left')
df.columns = [i.lower()  for i in df.columns]

##########################
# vectorize variables 
df = pd.get_dummies(df, columns = ['property_type','zip', 'mth', 'city'])

X = df.drop(columns = ['price', 'year', 'month'])
y = df['price']

# split data into training and testing
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape, X_val.shape
##############################################################################
# train XGBoost model 

# need to create function to return accuracy of GridSearchCV
def tuning_para(alg, tuning_para):
    gsearch = GridSearchCV(estimator=alg,
                           param_grid = tuning_para, 
                           scoring=make_scorer(mean_squared_error, squared=False, greater_is_better=False),
                           #scoring=make_scorer(r2_score),
                           n_jobs=-1, 
                           cv=5,
                           verbose = 2)
    
    gsearch.fit(X_train, y_train)
    print(gsearch.best_estimator_.get_params())    
    print('**************************************')
    for i, j in zip(gsearch.cv_results_['params'], gsearch.cv_results_['mean_test_score']):
        print(i, int(j) if abs(j)>1 else j)    
    print('**************************************')
    # predict test data 
    predictions = gsearch.predict(X_val)
    
    print('Mean-Absolute-Error(MAE): {}'.format(mean_absolute_error(y_val, predictions)))
    print('Root-Mean-Squared-Error(RMSE): {}'.format(mean_squared_error(y_val, predictions, squared = False)))
    print('Mean absolute percentage error (MAPE): {}'.format(mean_absolute_percentage_error(y_val, predictions)))
    print('R2: {}'.format(r2_score(y_val, predictions)))
    print([f'{key}: {gsearch.best_estimator_.get_params().get(key)}' for key in tuning_para.keys()])
    print('**************************************')
    return gsearch

#################################
# Step 0:
para_test = {}
xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state =27)
xgb_model_deploy = tuning_para(xgb_model, para_test)    
# Mean-Absolute-Error(MAE): 74369.610759325
# Root-Mean-Squared-Error(RMSE): 152160.3423311148
# Mean absolute percentage error (MAPE): 95.91093692796474
# R2: 0.9409015499525006
# []

#################################    
# step 1: let's test learning rate and n_estimator first
para_test = {'learning_rate':[0.01, 0.05, 0.1, 0.2],
             'n_estimators':[100, 200, 300]}
xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state =27)
xgb_model_deploy = tuning_para(xgb_model, para_test)


#################################
# step 2: Tune max_depth and min_child_weight
para_test = {
 'max_depth':range(3,10,2), # max depth of each tree
 'min_child_weight':range(1,6,2) # cover 
}

xgb_model = XGBRegressor(objective = 'reg:squarederror', 
                         random_state =27,
                         learning_rate = 0.1,
                         n_estimators = 300
                         )
xgb_model_deploy = tuning_para(xgb_model, para_test)

#################################
# Step 3: Tune gamma
para_test= {
    'gamma':[i/10.0 for i in range(0,5)] # the min gain requried to split
}
xgb_model = XGBRegressor(objective = 'reg:squarederror', 
                         random_state =27,
                         learning_rate = 0.1,
                         n_estimators = 300,
                         max_depth = 7,
                         min_child_weight = 1
                         )
xgb_model_deploy = tuning_para(xgb_model, para_test)

#################################
# Step 4: Tune subsample and colsample_bytree
para_test = {
 'subsample':[i/10.0 for i in range(6,10)],  # number of samples allowed 
 'colsample_bytree':[i/10.0 for i in range(6,10)] # number of fields allowed in each tree
}
xgb_model = XGBRegressor(objective = 'reg:squarederror', 
                         random_state =27,
                         learning_rate = 0.1,
                         n_estimators = 300,
                         max_depth = 7,
                         min_child_weight = 1,
                         gamma = 0
                         )
xgb_model_deploy = tuning_para(xgb_model, para_test)

#################################
# final model 
xgb_model = XGBRegressor(objective = 'reg:squarederror', 
                         random_state =27,
                         learning_rate = 0.1,
                         n_estimators = 300,
                         max_depth = 7,
                         min_child_weight = 1,
                         gamma = 0,
                         colsample_bytree = 0.7,
                         subsample = 0.8
                         )

xgb_model_deploy = xgb_model.fit(X_train.append(X_val), y_train.append(y_val))
prediction  = xgb_model_deploy.predict(X_train.append(X_val))

st_dev = (mean_squared_error(prediction, y_train.append(y_val)) ** 0.5 ).round(-3)
xgb_model_deploy.st_dev = st_dev

with open(r'\xgb_model_deploy.pickle', 'wb') as f:
    pickle.dump(xgb_model_deploy, f)
    
plt.rcParams["figure.figsize"] = (5,15)
plot_importance(xgb_model_deploy)
plt.show()





















































