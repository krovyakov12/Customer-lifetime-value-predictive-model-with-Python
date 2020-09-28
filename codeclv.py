%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifetimes
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


os.chdir(r"C:\quick_course\fg")

df = pd.read_csv("Online Retail.csv")
list(df.columns)

df.shape

df.dtypes

df.isnull()

df.isnull().sum()

df[df.CustomerID.isnull()].head()

135080/df.shape[0] # 0.2492    df.shape[0] = len(df)

df['CustomerID'].nunique()

#len(set(df['CustomerID'])) # slower

len(df[df.Quantity<0])

dfnew = df[(df.Quantity>0) & (df.CustomerID.isnull() == False)]

dfnew.shape


dfnew['amt'] = dfnew['Quantity'] * dfnew['UnitPrice']

dfnew['InvoiceDate'] = pd.to_datetime(dfnew['InvoiceDate']).dt.date








from lifetimes.plotting import *
from lifetimes.utils import *

modeldata = summary_data_from_transaction_data(dfnew, 'CustomerID', 'InvoiceDate', monetary_value_col='amt', observation_period_end='2011-12-9')
modeldata.head()

modeldata['frequency'].plot(kind='hist', bins=50)
print(modeldata['frequency'].describe())
print(sum(modeldata['frequency'] == 0)/float(len(modeldata)))

dfnew[dfnew.CustomerID == 12346.0]














from lifetimes import BetaGeoFitter
# similar API to scikit-learn and lifelines.

bgf = BetaGeoFitter(penalizer_coef=0.0)

bgf.fit(modeldata['frequency'], modeldata['recency'], modeldata['T'])
print(bgf)






























# create frequency recency matrix
from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)





















from lifetimes.plotting import plot_probability_alive_matrix

fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)






















t = 1
modeldata['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, modeldata['frequency'], modeldata['recency'], modeldata['T'])
modeldata.sort_values(by='predicted_purchases').tail(5)

modeldata.sort_values(by='predicted_purchases').head(895)

























from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)

























summary_cal_holdout = calibration_and_holdout_data(df, 'CustomerID', 'InvoiceDate',
                                        calibration_period_end='2011-06-08',
                                        observation_period_end='2011-12-9' )   
print(summary_cal_holdout.head())

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)





t = 10
individual = modeldata.loc[12380]
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])











'''
With the CLV model we have created,
we can specifically estimate a customer's historical probability of 
being alive. 

'''

from lifetimes.plotting import plot_history_alive
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
id = 14620  # id = 18074  id = 14606
days_since_birth = 365
sp_trans = df.loc[df['CustomerID'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')












# We are only estimating the customers who had at least one repeat purchase with us
returning_customers_summary = modeldata[modeldata['frequency']>0]

print(returning_customers_summary.head())

print(len(returning_customers_summary))
modeldata.shape






















from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])

print(ggf)

print(ggf.conditional_expected_average_profit(modeldata['frequency'], \
    modeldata['monetary_value']).head(10))


















##################use XGB###################

dfnew.InvoiceDate.max()
dfnew.InvoiceDate.min()
A = dfnew[dfnew.InvoiceDate>datetime.date(2011, 11, 9)]
A.shape
Z = pd.DataFrame(A.CustomerID.value_counts()).reset_index()
list(Z.columns)
Z.columns = ['CustomerID','buytime']
Z = Z.sort_values(['buytime'], ascending = False)
Z.head(20)

len(Z)
len(set(dfnew.CustomerID))

allcus = pd.DataFrame(set(dfnew.CustomerID), columns = ['CustomerID'])

allcus = pd.merge(allcus, Z, on=['CustomerID'], how='left').sort_values(['buytime'], ascending = False)
allcus = allcus.fillna(0)

allcus['buytime'] = allcus['buytime']/allcus['buytime'].max()

allcus['v1'] = allcus['buytime']*0.78 + np.random.normal(0, 0.11, len(allcus))

allcus['v2'] = allcus['buytime']*allcus['buytime']*(-0.195) + 0.11*allcus['buytime']

allcus.corr()


allcus['v1'] = (allcus['v1'] - allcus['v1'].min())/(allcus['v1'].max() - allcus['v1'].min())

allcus['v2'] = (allcus['v2'] - allcus['v2'].min())/(allcus['v2'].max() - allcus['v2'].min())

allcus['score'] = round(allcus['v2']*100000)
allcus.score.max()


allcus['discount'] = round(allcus['v1']*12.6)

allcus['r'] = np.random.randint(100,size=len(allcus))

def def2(D):
    if D['r']<13:
        x = 0
    else:
        x = D['discount']
    
    return x

allcus['discount'] = allcus.apply(def2, axis = 1)

v = ['CustomerID', 'score', 'discount']
allcus = allcus[v]

allcus = allcus.sort_values(['score'])

allcus.score.std()
allcus.corr()

allcus.to_csv('oth.csv', index = False)
####################################################



import datetime

list(dfnew.columns)
dfnew_train = dfnew[dfnew.InvoiceDate < datetime.date(2011, 11, 9)]
dfnew_test = dfnew[dfnew.InvoiceDate >= datetime.date(2011, 11, 9)]

maxdate = dfnew_train.InvoiceDate.max()
mindate = dfnew_train.InvoiceDate.min()

dfnew_train['duration'] =  (maxdate - dfnew_train.InvoiceDate)/np.timedelta64(1, 'D')
# get time duration between the last transaction to now
dfsum1 = dfnew_train.groupby(['CustomerID'])['duration'].min().reset_index()
dfsum1.head()

dfsum1.rename(columns = {'duration':'latetime'}, inplace = True)

# get time duration between the first transaction to now
dfsum2 = dfnew_train.groupby(['CustomerID'])['duration'].max().reset_index()
dfsum2.rename(columns = {'duration':'earlytime'}, inplace = True)

# get transaction frequency (whole history)
dfnew_train['freq'] =1 
dfsum3 = dfnew_train.groupby(['CustomerID'])['freq'].sum().reset_index()

# get transaction frequency (recent 3 months history)
dfnew_train['freq_3m'] =1 
dfsum4 = dfnew_train[dfnew_train.duration<91].groupby(['CustomerID'])['freq_3m'].sum().reset_index()

dfsum = pd.merge(dfsum1, dfsum2, on=['CustomerID'], how='outer')
dfsum = pd.merge(dfsum, dfsum3, on=['CustomerID'], how='outer')
dfsum = pd.merge(dfsum, dfsum4, on=['CustomerID'], how='outer')

#get other data source
other_data = pd.read_csv('oth.csv')
list(other_data.columns)
dfsum = pd.merge(dfsum, other_data, on=['CustomerID'], how='left')

# get target 
dfnew_test['target'] = 1
dfsum_target = dfnew_test.groupby(['CustomerID'])['target'].sum().reset_index()

dfsum = pd.merge(dfsum, dfsum_target, on=['CustomerID'], how='left')
dfsum  = dfsum.fillna(0).sort_values(['target'], ascending = False)

# check all features in the modeling data
list(dfsum.columns)
dfsum.head(10)

####################xgb model####################################
import xgboost
from sklearn.model_selection import train_test_split

xgb_model = xgboost.XGBRegressor(n_estimators=2200, objective='reg:linear', max_depth = 5)
 
predictors = ['latetime','earlytime', 'freq','freq_3m', 'score','discount']
X = dfsum[predictors]
y = dfsum.target

x_trains, x_valids, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=12)

xgb_model.fit(x_trains, y_train)

preds = xgb_model.predict(x_valids)
errs= np.abs(preds - y_valid)**2
mse = np.sqrt(errs.mean())

from xgboost import plot_importance
from matplotlib import pyplot



# for XGBRegressor, we use the following ways
xgb_model.get_booster().get_score(importance_type="gain")
xgb_model.get_booster().get_score(importance_type="weight")

'''
importance_type:
    
‘weight’ - the number of times a feature is used to split the data across all trees.
‘gain’ - the average gain across all splits the feature is used in.
‘cover’ - the average coverage across all splits the feature is used in. (yes/no proportion)
‘total_gain’ - the total gain across all splits the feature is used in.
‘total_cover’ - the total coverage across all splits the feature is used in.

'''

# plot feature importance
plot_importance(xgb_model, importance_type='gain')
pyplot.show()

plot_importance(xgb_model, importance_type='weight') # default option is 'weight'
pyplot.show()


# we can also use feature_importances_, but it is the same as importance_type='gain'
imp = xgb_model.feature_importances_
pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
pyplot.show()

#  we can list importance by dict, by 'gain'
sorted_idx = np.argsort(xgb_model.feature_importances_)[::-1]
for index in sorted_idx:
    print([x_trains.columns[index], xgb_model.feature_importances_[index]]) 

important_var = [(x_trains.columns[index], xgb_model.feature_importances_[index]) for index in sorted_idx]

# check correlation of features with target
dfsum.corr()







##########################light gbm model#######################################
import lightgbm as lgb   

lgbparams = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',    
    'max_depth': 6, 
    'learning_rate': 0.02
}

predictors = ['latetime','earlytime', 'freq','freq_3m', 'score','discount']  
  
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=12)
        
x_trains, x_valids, y_train, y_valid = train_test_split(X1, y1, test_size=0.1, random_state=12)
x_train = x_trains[predictors]
x_valid = x_valids[predictors]

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

# for monitoring the performance
watchlist = [d_valid]
n_estimators = 2200 # this para will be set in train()


model = lgb.train(lgbparams, d_train, n_estimators, watchlist, verbose_eval=1)

preds = model.predict(X2)
errs= np.abs(preds - y2)**2
mse = np.sqrt(errs.mean())

print('Feature importances:', list(model.feature_importance()))
important_var = list(zip(predictors,list(model.feature_importance())))
important_var.sort(key = lambda t: t[1])

print (important_var)
len(important_var)

import matplotlib.pyplot as plt
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(), x_train.columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))

# other way seaborn for importance levels
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()





