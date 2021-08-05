<div class="section-content"><div class="section-inner sectionLayout--insetColumn"><h3 name="e752" class="graf graf--h3 graf--leading graf--title">Create a Home Price Prediction App with Plotly-Dash, Google BigQuery and Machine&nbsp;Learning</h3><h4 name="e064" class="graf graf--h4 graf-after--h3 graf--subtitle">Predict home price with XGBoost using sales data from redfin.com</h4><figure tabindex="0" name="0277" class="graf graf--figure graf-after--h4 is-selected" contenteditable="false"><div class="aspectRatioPlaceholder is-locked" style="max-width: 700px; max-height: 526px;"><div class="aspectRatioPlaceholder-fill" style="padding-bottom: 75.1%;"></div><img class="graf-image" data-image-id="0*qwI_WMY9_1z-S84b" data-width="4592" data-height="3448" data-unsplash-photo-id="rgJ1J8SDEAY" src="https://cdn-images-1.medium.com/max/800/0*qwI_WMY9_1z-S84b"><div class="crosshair u-ignoreBlock"></div></div><figcaption class="imageCaption" data-default-value="Type caption for image (optional)" contenteditable="true">Photo by <a href="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%2F%40tierramallorca%3Futm_source%3Dmedium%26utm_medium%3Dreferral" data-href="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%2F%40tierramallorca%3Futm_source%3Dmedium%26utm_medium%3Dreferral" class="markup--anchor markup--figure-anchor" rel="photo-creator" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%2F%40tierramallorca%3Futm_source%3Dmedium%26utm_medium%3Dreferral" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">Tierra Mallorca</a> on&nbsp;<a href="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%3Futm_source%3Dmedium%26utm_medium%3Dreferral" data-href="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%3Futm_source%3Dmedium%26utm_medium%3Dreferral" class="markup--anchor markup--figure-anchor" rel="photo-source" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Funsplash.com%3Futm_source%3Dmedium%26utm_medium%3Dreferral" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">Unsplash</a></figcaption></figure><p name="cb95" class="graf graf--p graf-after--figure">Check out the <a href="https://medium.com/r/?url=https%3A%2F%2Fdsprojectapp.herokuapp.com%2F" data-href="https://medium.com/r/?url=https%3A%2F%2Fdsprojectapp.herokuapp.com%2F" class="markup--anchor markup--p-anchor" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Fdsprojectapp.herokuapp.com%2F" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">Web App</a> and the code on my <a href="https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Faaronzhuclover%2FHome-Price-Prediction" data-href="https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Faaronzhuclover%2FHome-Price-Prediction" class="markup--anchor markup--p-anchor" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Faaronzhuclover%2FHome-Price-Prediction" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">GitHub</a> and feel free to let me know if you have any questions!</p><h3 name="4b5e" class="graf graf--h3 graf-after--p"><strong class="markup--strong markup--h3-strong">Project Background</strong></h3><p name="19bf" class="graf graf--p graf-after--h3">Whether you’re buying or selling a house, change in home prices will affect your housing plans. Keeping an eye on home prices can give you an idea of what to expect if you plan to buy or sell a house any time soon.&nbsp;</p><p name="599e" class="graf graf--p graf-after--p">Therefore, in this project, I would like to utilize my data science skills to create an interesting app that is capable of keeping track of home price, sales volume and making price prediction using Machine Learning. To make it more interesting, it also tell users if the price of a newly listed house is high or low. [instead of priec -diff, return price percentile.]</p><h3 name="c0aa" class="graf graf--h3 graf-after--p"><strong class="markup--strong markup--h3-strong">Back-End: Building Data&nbsp;Pipeline</strong></h3><p name="db9e" class="graf graf--p graf-after--h3"><strong class="markup--strong markup--p-strong">Collect Sales Data from redfin.com</strong></p><p name="ec59" class="graf graf--p graf-after--p">For this project, I wrote a Python script to download home sales data from redfin.com. The script will run automatically on a daily basis using Task Scheduler in a windows machine.</p><figure tabindex="0" name="2ba0" class="graf graf--figure graf-after--p" contenteditable="false"><div class="aspectRatioPlaceholder is-locked" style="max-width: 700px; max-height: 384px;"><div class="aspectRatioPlaceholder-fill" style="padding-bottom: 54.900000000000006%;"></div><img class="graf-image" data-image-id="0*OPtRNr5JrtvLRMWH" data-width="1600" data-height="878" src="https://cdn-images-1.medium.com/max/800/0*OPtRNr5JrtvLRMWH"><div class="crosshair u-ignoreBlock"></div></div><figcaption class="imageCaption" data-default-value="Type caption for image (optional)" contenteditable="true">redfin.com</figcaption></figure><p name="628c" class="graf graf--p graf-after--figure">On a given redfin page, we can copy the download link, which would be something similar to the following. We can twist the parameters, such as, “num_homes”. “region_id” and “sf” in the URL to download the data sets we are interested in.&nbsp;</p><pre name="a456" class="graf graf--pre graf-after--p"><a href="https://medium.com/r/?url=https%3A%2F%2Fwww.redfin.com%2Fstingray%2Fapi%2Fgis-csv%3Fal%3D1%26market%3Dsocal%26min_stories%3D1%26num_homes%3D350%26ord%3Dredfin-recommended-asc%26page_number%3D1%26region_id%3D25415%26region_type%3D6%26sf%3D1%2C2%2C3%2C5%2C6%2C7%26status%3D9%26uipt%3D1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%26v%3D8" data-href="https://medium.com/r/?url=https%3A%2F%2Fwww.redfin.com%2Fstingray%2Fapi%2Fgis-csv%3Fal%3D1%26market%3Dsocal%26min_stories%3D1%26num_homes%3D350%26ord%3Dredfin-recommended-asc%26page_number%3D1%26region_id%3D25415%26region_type%3D6%26sf%3D1%2C2%2C3%2C5%2C6%2C7%26status%3D9%26uipt%3D1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%26v%3D8" class="markup--anchor markup--pre-anchor" target="_blank">https://www.redfin.com/stingray/api/gis-csv?al=1&amp;market=socal&amp;min_stories=1&amp;num_homes=350&amp;ord=redfin-recommended-asc&amp;page_number=1&amp;region_id=25415&amp;region_type=6&amp;sf=1,2,3,5,6,7&amp;status=9&amp;uipt=1,2,3,4,5,6,7,8&amp;v=8</a></pre><p name="64a2" class="graf graf--p graf-after--pre">Downloading the home sales data in CSV format from redfin is straightforward using “requests” library in Python.&nbsp;</p><pre name="4821" class="graf graf--pre graf-after--p">import pandas as pd<br>import requests</pre><pre name="b808" class="graf graf--pre graf-after--pre">url = r’https://www.redfin.com/stingray/api/gis-csv?al=1&amp;market=socal&amp;min_stories=1&amp;num_homes=350&amp;ord=redfin-recommended-asc&amp;page_number=1&amp;region_id=25415&amp;region_type=6&amp;sf=1,2,3,5,6,7&amp;status=9&amp;uipt=1,2,3,4,5,6,7,8&amp;v=8’</pre><pre name="cbb6" class="graf graf--pre graf-after--pre">file = requests.get(url, headers={‘User-Agent’: ‘Mozilla/5.0’}).content</pre><pre name="4414" class="graf graf--pre graf-after--pre">df = pd.read_csv(io.StringIO(file.decode(‘utf-8’)))</pre><p name="d62d" class="graf graf--p graf-after--pre"><strong class="markup--strong markup--p-strong">Redfin API</strong></p><p name="9dcd" class="graf graf--p graf-after--p">Besides home sales data, we might also include other explanatory variables that control individual home effect in our ML models. <a href="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Fredfin%2F" data-href="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Fredfin%2F" class="markup--anchor markup--p-anchor" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Fredfin%2F" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">Redfin API</a> provides additional data for a property listed in redfin.com, such as, School Rating, Walk Score, Transit Score and Bike Score.</p><pre name="d51d" class="graf graf--pre graf-after--p">from redfin import Redfin</pre><pre name="ef29" class="graf graf--pre graf-after--pre">client = Redfin()<br>address = ‘628 Castlehill Dr, Walnut, CA 91789’<br>response = client.search(address)<br>url = response[‘payload’][‘exactMatch’][‘url’]<br>initial_info = client.initial_info(url)<br>property_id = initial_info[‘payload’][‘propertyId’]<br>listing_id = initial_info[‘payload’][‘listingId’]<br>mls_data = client.below_the_fold(property_id)<br>schools_rating = mls_data[‘payload’][‘schoolsAndDistrictsInfo’][‘servingThisHomeSchools’]</pre><p name="f77b" class="graf graf--p graf-after--pre"><strong class="markup--strong markup--p-strong">FRED API</strong></p><p name="feab" class="graf graf--p graf-after--p">Macro-economic variables that control market effects are also important to include in the ML models. <a href="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Ffredapi%2F" data-href="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Ffredapi%2F" class="markup--anchor markup--p-anchor" data-tooltip="https://medium.com/r/?url=https%3A%2F%2Fpypi.org%2Fproject%2Ffredapi%2F" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">FRED API</a> comes in handy to import these external economic factors. For this project, I used two economic variables, S&amp;P/Case-Shiller CA-Los Angeles Home Price Index and 30-Year Fixed Rate Mortgage Rate.</p><pre name="6e81" class="graf graf--pre graf-after--p">from fredapi import Fred<br>fred = Fred(api_key='3e45db934f364bc329aca420c85fa04e')<br># extract S&amp;P/Case-Shiller CA-Los Angeles Home Price Index (LXXRNSA)<br># <a href="https://medium.com/r/?url=https%3A%2F%2Ffred.stlouisfed.org%2Fseries%2FLXXRNSA" data-href="https://medium.com/r/?url=https%3A%2F%2Ffred.stlouisfed.org%2Fseries%2FLXXRNSA" class="markup--anchor markup--pre-anchor" rel="nofollow" target="_blank">https://fred.stlouisfed.org/series/LXXRNSA</a><br>la_hpi_raw = fred.get_series('LXXRNSA')<br>la_hpi = la_hpi_raw.to_frame()<br>la_hpi.columns = ['hpi']<br>la_hpi['month'] = la_hpi.index<br>la_hpi = la_hpi.reset_index(drop = True)<br>la_hpi['month'] = la_hpi.apply(lambda x: x['month'].date(), axis = 1)<br>weights = np.array([0.2, 0.3, 0.5])<br>sum_weights = np.sum(weights)<br># compute weighted MA from latest 3 months<br>la_hpi['hpi'] = la_hpi['hpi'].rolling(3).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False).reset_index(drop = True)<br>month_diff = datetime.now().month - la_hpi['month'].max().month<br># reset the month variable so that it can be merged with sales data<br>la_hpi['month'] = la_hpi['month'] + pd.DateOffset(months=month_diff)<br>la_hpi['month'] = la_hpi.apply(lambda x: x['month'].date(), axis = 1)<br><br># 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US)<br># <a href="https://medium.com/r/?url=https%3A%2F%2Ffred.stlouisfed.org%2Fseries%2FMORTGAGE30US" data-href="https://medium.com/r/?url=https%3A%2F%2Ffred.stlouisfed.org%2Fseries%2FMORTGAGE30US" class="markup--anchor markup--pre-anchor" rel="nofollow" target="_blank">https://fred.stlouisfed.org/series/MORTGAGE30US</a><br>mort_rate_30yrs_raw = fred.get_series('MORTGAGE30US')<br>mort_rate_30yrs = mort_rate_30yrs_raw.to_frame()<br>mort_rate_30yrs.columns = ['mort_rate']<br>mort_rate_30yrs['date'] = mort_rate_30yrs.index<br>mort_rate_30yrs = mort_rate_30yrs.reset_index(drop = True)<br>mort_rate_30yrs['date'] = mort_rate_30yrs.apply(lambda x: x['date'].date(), axis = 1)<br>mort_rate_30yrs['year'] = mort_rate_30yrs.apply(lambda x: x['date'].year, axis = 1)<br>mort_rate_30yrs['month'] = mort_rate_30yrs.apply(lambda x: x['date'].month, axis = 1)<br>mort_rate_30yrs = mort_rate_30yrs.groupby(['year', 'month'])['mort_rate'].mean().reset_index()<br>mort_rate_30yrs['month'] = mort_rate_30yrs.apply(lambda x: date(int(x['year']), int(x['month']), 1), axis = 1)<br>mort_rate_30yrs = mort_rate_30yrs.drop(columns = ['year'])</pre><p name="dd7d" class="graf graf--p graf-after--pre"><strong class="markup--strong markup--p-strong">Data cleaning using Pandas</strong></p><p name="7251" class="graf graf--p graf-after--p">Once I collect all the data we needed for this project, I use the popular data manipulation library, “Pandas” to clean, consolidate and aggregate the data.</p><p name="2d04" class="graf graf--p graf-after--p">In the process, I removed data with missing “Price”, “Sold Date” and kept the sales data from cities that I am interested in. I also corrected the “Lot Size” for Condo/Co-op. The list of features I extracted from the sales data include “Number of Beds”, “Number of Baths”, “Square Feet”, “Lot Size”, “Age”, “HOA”, Dummies of “Property Type”, “Zip Code”, “City”, and “Month of Sale”.</p><p name="2199" class="graf graf--p graf-after--p">Economic variables might not be published in a timely manner. For example, S&amp;P/Case-Shiller CA-Los Angeles Home Price Index for May won’t be published until by the end of June. Therefore, it is not necessary to join sale data with economic variables in the same month. Instead, I computed weighted moving average of latest 3 months and joined with sale data in the current month.  &nbsp;</p><pre name="d9aa" class="graf graf--pre graf-after--p">import pandas as pd</pre><pre name="0a60" class="graf graf--pre graf-after--pre">data.rename(columns = {'URL (SEE <a href="https://medium.com/r/?url=http%3A%2F%2Fwww.redfin.com%2Fbuy-a-home%2Fcomparative-market-analysis" data-href="https://medium.com/r/?url=http%3A%2F%2Fwww.redfin.com%2Fbuy-a-home%2Fcomparative-market-analysis" class="markup--anchor markup--pre-anchor" rel="nofollow" target="_blank">http://www.redfin.com/buy-a-home/comparative-market-analysis</a> FOR INFO ON PRICING)':'URL'}, inplace = True)<br>data = data[~data['CITY'].isnull()]<br>data = data[data['CITY'].isin(['Irvine', 'Arcadia', 'El Monte', 'Walnut', 'Rowland Heights'])]<br>data = data[data['PROPERTY TYPE'].str.contains('Single|Condo|Townhouse')]<br>data = data[~data['SOLD DATE'].isnull()]<br>data.rename(columns = {'ZIP OR POSTAL CODE': 'ZIP',<br>                          'HOA/MONTH': 'HOA',<br>                          'PROPERTY TYPE': 'PROPERTY_TYPE',<br>                          'SQUARE FEET': 'SQUARE_FEET',<br>                          'LOT SIZE': 'LOT_SIZE'<br>                          }, inplace = True)<br>data = data[~data['PRICE'].isnull()]<br>data['PRICE'] = data['PRICE'].astype('int')<br># SOLD DATE: month<br>data['SOLD DATE2'] = pd.to_datetime(data['SOLD DATE'], format='%B-%d-%Y')<br>data['year'] = pd.DatetimeIndex(data['SOLD DATE2']).year<br>data['mth'] = pd.DatetimeIndex(data['SOLD DATE2']).month<br>data['month'] = data.apply(lambda x: date(int(x['year']), int(x['mth']), 1), axis = 1)<br># YEAR BUILT: age<br>data['age'] = data['year'] - data['YEAR BUILT']<br># HOA/MONTH: if nan, set it to be $0<br>data['HOA'] = data.apply(lambda x: 0 if (np.isnan(x['HOA']) &amp; bool(re.findall('Single', x['PROPERTY_TYPE'])))  else  x['HOA'], axis =1)<br># LOT SIZE: change LOT SIZE to be  SQUARE FEET for non-single house<br>data['LOT_SIZE'] = data.apply(lambda x: x['LOT_SIZE'] if re.findall('Single', x['PROPERTY_TYPE']) else  x['SQUARE_FEET']    , axis =1)<br># ZIP<br>data = data[~data['ZIP'].isnull()]<br>data['ZIP'] = data['ZIP'].astype('int')<br>zip_keep = data['ZIP'].value_counts()<br>zip_keep = zip_keep[zip_keep/len(data) &gt; 0.01].index<br>data = data[data['ZIP'].isin(zip_keep)]<br>data['ZIP'].value_counts(dropna = False)<br>keep_var = ['PRICE', 'PROPERTY_TYPE', 'CITY', 'ZIP', 'BEDS', 'BATHS', 'SQUARE_FEET', 'LOT_SIZE', 'age', 'HOA', 'year', 'mth', 'month']<br>data = data[keep_var]<br>df = data.merge(mort_rate_30yrs, on = 'month', how = 'left')<br>df = df.merge(la_hpi, on = 'month', how = 'left')<br>df.columns = [i.lower()  for i in df.columns]</pre><p name="06d6" class="graf graf--p graf-after--pre"><strong class="markup--strong markup--p-strong">Store data in Google BigQuery</strong></p><p name="c48d" class="graf graf--p graf-after--p">Once data are cleaned and pre-processed, data will be stored in Google BigQuery and the Python script would automatically update the SQL database on a daily basis so that my application will access to the latest back-end data.</p><pre name="54a3" class="graf graf--pre graf-after--p">from google.cloud import bigquery<br>from google.oauth2 import service_account<br>import json<br>import tempfile<br>from pandas_gbq import schema<br>project = 'testpython-267102'<br>credentials = service_account.Credentials.from_service_account_file(r'google_creds.json')<br>client = bigquery.Client(credentials= credentials, project=project)<br>sql = '''<br> DROP TABLE IF EXISTS `sampledata.sales_data`;<br> CREATE TABLE `sampledata.sales_data`(<br>  property_type string,<br>  address string,<br>  city string,<br>  zip FLOAT64,<br>  price INT64,<br>  beds FLOAT64,<br>  baths FLOAT64,<br>  sq_feet FLOAT64,<br>  lot_size FLOAT64,<br>  year_built FLOAT64<br>  hoa FLOAT64<br>  sold_date string<br> );<br> '''<br>query = client.query(sql)<br># insert using pd.to_gbq<br># print(schema.generate_bq_schema(master))<br>schema_ = schema.generate_bq_schema(data)['fields']<br>data.to_gbq(destination_table = 'sampledata.sales_data',<br>  project_id = project,<br>  if_exists = 'append',<br>  credentials = credentials,<br>  table_schema=schema_<br>)</pre><h3 name="f3ce" class="graf graf--h3 graf-after--pre"><strong class="markup--strong markup--h3-strong">Modeling with&nbsp;XGBoost</strong></h3><p name="f2c7" class="graf graf--p graf-after--h3">Now we come to the modeling part. In this project, I used a decision-tree-based algorithm, <strong class="markup--strong markup--p-strong">XGBoost</strong>. XGBoost provides an efficient implementation of gradient boosting that can be used for regression predictive modeling. Trees are added on at a time to correct the prediction errors made from prior models.</p><p name="51f4" class="graf graf--p graf-after--p">To tune hyperparameters, <strong class="markup--strong markup--p-strong">Grid Search</strong> was applied throughout the modeling process to get best set of parameters.</p><pre name="2ce9" class="graf graf--pre graf-after--p">from sklearn.model_selection import GridSearchCV<br>from xgboost import XGBRegressor<br>from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score, mean_absolute_percentage_error</pre><pre name="4b0d" class="graf graf--pre graf-after--pre"># split data into training and testing<br>X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=0)<br>X_train.shape, X_val.shape<br>##############################################################################<br># train XGBoost model</pre><pre name="a8f1" class="graf graf--pre graf-after--pre"># need to create function to return accuracy of GridSearchCV<br>def tuning_para(alg, tuning_para):<br>    gsearch = GridSearchCV(estimator=alg,<br>                           param_grid = tuning_para, <br>                           scoring=make_scorer(mean_squared_error, squared=False, greater_is_better=False),<br>                           #scoring=make_scorer(r2_score),<br>                           n_jobs=-1, <br>                           cv=5,<br>                           verbose = 2)   <br>    gsearch.fit(X_train, y_train)<br>    predictions = gsearch.predict(X_val)<br>    return gsearch</pre><pre name="417e" class="graf graf--pre graf-after--pre">#################################    <br># step 1: let's test learning rate and n_estimator first<br>para_test = {'learning_rate':[0.01, 0.05, 0.1, 0.2],<br>             'n_estimators':[100, 200, 300]}<br>xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state =27)<br>xgb_model_deploy = tuning_para(xgb_model, para_test)<br>#################################<br># step 2: Tune max_depth and min_child_weight<br>para_test = {<br> 'max_depth':range(3,10,2), # max depth of each tree<br> 'min_child_weight':range(1,6,2) # cover <br>}</pre><pre name="8354" class="graf graf--pre graf-after--pre">xgb_model = XGBRegressor(objective = 'reg:squarederror', <br>                         random_state =27,<br>                         learning_rate = 0.05,<br>                         n_estimators = 300<br>                         )<br>xgb_model_deploy = tuning_para(xgb_model, para_test)<br>#################################<br># Step 3: Tune gamma<br>para_test= {<br>    'gamma':[i/10.0 for i in range(0,5)] # the min gain requried to split<br>}<br>xgb_model = XGBRegressor(objective = 'reg:squarederror', <br>                         random_state =27,<br>                         learning_rate = 0.05,<br>                         n_estimators = 300,<br>                         max_depth = 7,<br>                         min_child_weight = 1<br>                         )<br>xgb_model_deploy = tuning_para(xgb_model, para_test)<br>#################################<br># Step 4: Tune subsample and colsample_bytree<br>para_test = {<br> 'subsample':[i/10.0 for i in range(6,10)],  # number of samples allowed <br> 'colsample_bytree':[i/10.0 for i in range(6,10)] # number of fields allowed in each tree<br>}<br>xgb_model = XGBRegressor(objective = 'reg:squarederror', <br>                         random_state =27,<br>                         learning_rate = 0.05,<br>                         n_estimators = 300,<br>                         max_depth = 7,<br>                         min_child_weight = 1,<br>                         gamma = 0<br>                         )<br>xgb_model_deploy = tuning_para(xgb_model, para_test)<br>#################################<br># final model <br>xgb_model = XGBRegressor(objective = 'reg:squarederror', <br>                         random_state =27,<br>                         learning_rate = 0.05,<br>                         n_estimators = 300,<br>                         max_depth = 7,<br>                         min_child_weight = 1,<br>                         gamma = 0,<br>                         colsample_bytree = 0.8,<br>                         subsample = 0.9<br>                         )</pre><p name="d0dc" class="graf graf--p graf-after--pre">In regression problems, machine learning models, such as, XGBoost would predict a single value without giving a certainty of that value. Sometimes, it is useful to be able to measure <strong class="markup--strong markup--p-strong">the certainty level of a prediction</strong>. Especially, for the home price prediction model, I didn’t include individual house features, such as, swimming pool, solar panel, kitchen remodel, new roof, new carpet, etc.&nbsp;</p><p name="4e0d" class="graf graf--p graf-after--p">To compute the <strong class="markup--strong markup--p-strong">prediction interval</strong>, we need to product a prediction and an estimate error for that prediction. To simplify, I made an assumption that the desired prediction follows a normal distribution and the standard deviation of the normal distribution is constant. Therefore, we can use RMSE to estimate the standard deviation. (In practice, the error is not always constant. I also includes an improved model in the Final note)</p><pre name="3308" class="graf graf--pre graf-after--p">xgb_model_deploy = xgb_model.fit(X_train.append(X_val), y_train.append(y_val))<br>prediction  = xgb_model_deploy.predict(X_train.append(X_val))<br>st_dev = (mean_squared_error(prediction, y_train.append(y_val)) ** 0.5 ).round(-3)</pre><p name="6fbe" class="graf graf--p graf-after--pre">We can compute 90% prediction interval as [prediction-1.64*SD, prediction+1.64*SD].</p><figure tabindex="0" name="f7b1" class="graf graf--figure graf-after--p" contenteditable="false"><div class="aspectRatioPlaceholder is-locked" style="max-width: 700px; max-height: 251px;"><div class="aspectRatioPlaceholder-fill" style="padding-bottom: 35.9%;"></div><img class="graf-image" data-image-id="1*EPeZJQpw-jjMESpO66DT2A.png" data-width="1580" data-height="567" src="https://cdn-images-1.medium.com/max/800/1*EPeZJQpw-jjMESpO66DT2A.png" data-delayed-src="https://cdn-images-1.medium.com/max/800/1*EPeZJQpw-jjMESpO66DT2A.png"><div class="crosshair u-ignoreBlock"></div></div><figcaption class="imageCaption" data-default-value="Type caption for image (optional)" contenteditable="true">Prediction Interval</figcaption></figure><h3 name="e13a" class="graf graf--h3 graf-after--figure">Front-End: Creating a Web Application</h3><p name="7bbe" class="graf graf--p graf-after--h3"><strong class="markup--strong markup--p-strong">Create Dash App</strong></p><p name="17aa" class="graf graf--p graf-after--p">Creating a web application has become easy with the help of “Dash”. <strong class="markup--strong markup--p-strong">Dash</strong> give data scientists the ability to showcase their analysis in interactive web applications and expands the notion of what’s possible in a traditional “dashboard”. It uses Dash core components, HTML components, Bootstrap components and Callbacks to build interactive applications.</p><p name="5cf0" class="graf graf--p graf-after--p">In this project, I pre-processed data and pre-trained the model in a local windows machine. Therefore, the Dash app is very light-weighted and fast. It only needs to request and display the data from Google Bigquery.</p><pre name="d74a" class="graf graf--pre graf-after--p">import dash<br>import dash_core_components as dcc<br>import dash_html_components as html<br>from dash.dependencies import Input, Output, State<br>import dash_bootstrap_components as dbc</pre><pre name="0ddf" class="graf graf--pre graf-after--pre">external_stylesheets = [dbc.themes.BOOTSTRAP]<br>app = dash.Dash(__name__, external_stylesheets=external_stylesheets)<br>app.title = 'Housing Price Dashboard'<br>server = app.server<br>app.layout = html.Div([<br>dcc.Tabs(style = {'width': '100%'}, children=[<br>        dcc.Tab(label='Housing Price Index', children = [<br>        html.Div([<br>        html.Br(),<br>        dbc.Row([html.Div(children='Choose Year Range', style = {"margin-left": "30px"})]),<br>        dbc.Row([<br>            dbc.Col(<br>            dcc.RangeSlider(<br>            id='year-range-slider',<br>            min=2000,<br>            max=2021,<br>            step=1,<br>            value=[2020, 2021],<br>            marks = {i: str(i) for i in range(2000,2022, 1)}<br>            )<br>            )<br>        ]),<br>        html.Br(),<br>        dbc.Row([<br>        dbc.Col(<br>            dbc.Checklist(<br>            id="checklist",<br>            options=[{"label": x, "value": x} for x in all_locations],<br>            value=all_locations,<br>            labelStyle={'display': 'inline-block'},<br>            labelCheckedStyle={"color": "red"},<br>            inline=True, # arrange list horizontally<br>            style={"justify-content":"space-between", "font-size":"24px", "margin-left": "100px"}<br>            )<br>        )]),<br>        dbc.Row([<br>            dbc.Col(html.Div(dcc.Graph(id='line-graph'), style = {'width': '100%'}))<br>        ])<br>        ])<br>        ])<br>    ], style = {'padding': '20px'})<br><br># create callback for line graph<br><a href="https://medium.com/r/?url=http%3A%2F%2Ftwitter.com%2Fapp" data-href="https://medium.com/r/?url=http%3A%2F%2Ftwitter.com%2Fapp" class="markup--anchor markup--pre-anchor" title="Twitter profile for @app" data-tooltip="https://medium.com/r/?url=http%3A%2F%2Ftwitter.com%2Fapp" data-tooltip-position="bottom" data-tooltip-type="link" target="_blank">@app</a>.callback(<br>Output('line-graph', 'figure'),<br>Input('checklist', 'value'),<br>Input('year-range-slider', 'value')<br>)<br>def update_line_graph (cities, year_range):<br>    selected_df = df[df.location.isin(cities)]<br>    selected_df = selected_df.query(f'year&gt;={year_range[0]} &amp; year&lt;={year_range[1]}')<br>    fig = px.line(selected_df, x="month", y="price_med_ma6", color = 'location', title='6-Month Weighted Moving Average of Median Housing Price',<br>    labels = {'price_med_ma6':'Media Housing Pirce ($/SF)', 'location': 'City', 'month': ''}, height=500<br>)<br># edit hover effects<br>fig.update_traces(mode="lines", hovertemplate=None)<br># update figure layout<br>fig.update_layout(<br>    title = {'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 20}},<br>    legend = {'orientation': 'h', 'yanchor': "top", 'xanchor': "left", 'x': 0, 'font': {'size': 20} },<br>    legend_title='',<br>    hovermode="x unified",<br>    hoverlabel = {'font_size': 12, 'font_family': "Rockwell", 'namelength': 20}<br>)<br>return fig</pre><pre name="209a" class="graf graf--pre graf-after--pre"># run the app<br>if __name__ == '__main__':<br>app.run_server(debug=True)</pre><p name="ecc8" class="graf graf--p graf-after--pre"><strong class="markup--strong markup--p-strong">App Deployment with GitHub and Heroku</strong></p><p name="f9c7" class="graf graf--p graf-after--p"><strong class="markup--strong markup--p-strong">Heroku </strong>integrates with <strong class="markup--strong markup--p-strong">GitHub </strong>to make it easier to deploy codes on GitHub to apps hosted on Heroku. When we make changes on the codes on GitHub, it will automatically deploy updates on app on Heroku.</p><figure tabindex="0" name="45c0" class="graf graf--figure is-defaultValue graf-after--p" contenteditable="false"><div class="aspectRatioPlaceholder is-locked" style="max-width: 700px; max-height: 400px;"><div class="aspectRatioPlaceholder-fill" style="padding-bottom: 57.199999999999996%;"></div><img class="graf-image" data-image-id="1*2u6PLKWyIfoY_ySZI1OZuQ.png" data-width="1215" data-height="695" src="https://cdn-images-1.medium.com/max/800/1*2u6PLKWyIfoY_ySZI1OZuQ.png" data-delayed-src="https://cdn-images-1.medium.com/max/800/1*2u6PLKWyIfoY_ySZI1OZuQ.png"><div class="crosshair u-ignoreBlock"></div></div><figcaption class="imageCaption" data-default-value="Type caption for image (optional)" contenteditable="true"><span class="defaultValue">Type caption for image (optional)</span><br></figcaption></figure><h3 name="c1d0" class="graf graf--h3 graf-after--figure">Final Notes</h3><ul class="postList"><li name="7854" class="graf graf--li graf-after--h3">In practice, the error is not always constant. To improve the model, we can fit a model to forecast the error itself. The intuition is the home with more rooms, bigger square feet, or better neighborhood tend to have more price variation.&nbsp;</li><li name="e139" class="graf graf--li graf-after--li">Redfin API also provides text description of a property. It would include keywords, such as, “quiet neighborhood”, “newly remodeled kitchen”, “freshly painted exterior”, that can explain the home price. For future improvement, I can use NLP model, such as, Word2Vec to convert text into features in the ML models.</li></ul><p name="b722" class="graf graf--p graf-after--li graf--trailing"><strong class="markup--strong markup--p-strong">Thanks for reading!!!</strong></p></div></div>
