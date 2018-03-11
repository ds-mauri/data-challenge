""" Main goals:
1. Parse and extract the data.
2. Identify a pattern on any set of fields that can help predict how much a
customer will spend.
3. Calculate a sales forecast for the next week.

"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# Let's just use pandas since the data is small. It has a nice visualization
# in Spyder IDE.
df = pd.read_json('sample.txt')

# %%
# First things first. As long as one of the objectives is to calculate the
# sales forecast for the next month, let's plot how the sales varies along
# the days

# Let's create columns to help us work with the data more directly
df['amountSpent'] = pd.Series([df['complemento'].iloc[i]['valorTotal'] for i
                               in range(0, df.shape[0])])
df['date'] = pd.Series([df['ide'].iloc[i]['dhEmi']['$date'] for i
                        in range(0, df.shape[0])])
df['date'] = pd.to_datetime(df['date'])
df['hour'] =  pd.Series([df['date'].iloc[i].hour for i
                        in range(0, df.shape[0])])
df['day'] =  pd.Series([df['date'].iloc[i].day for i
                        in range(0, df.shape[0])])
df['week'] =  pd.Series([df['date'].iloc[i].week for i
                         in range(0, df.shape[0])])
df['weekday'] =  pd.Series([df['date'].iloc[i].weekday() for i
                             in range(0, df.shape[0])])

# We'll summarize the total spent by day in a pivot table
total_by_day = pd.pivot_table(data=df, values='amountSpent', index='day',
                              aggfunc='sum')
total_by_day = total_by_day.reset_index()

# Plot of the days vs. amount spent that day
plt.plot(total_by_day['day'], total_by_day['amountSpent'])

# Nice. As we see the amount spent during the week is pretty periodic and
# more important: it's stationary! A fundamental property to work on time
# series.
# That said, a first guess on the sales forecast could be the mean of the
# amount spent by weekday.

mean_week_day = pd.pivot_table(data=df, values='amountSpent', index='weekday',
                               columns='week', aggfunc='sum')
mean_week_day['mean'] = np.nanmean(mean_week_day, axis=1)
mean_week_day['day'] = pd.Series(range(25,31))  # The restaurante is closed on
# Sundays
# Let's artificially input the last day we have register on our forecast just
# to make the plot beautiful
mean_week_day = mean_week_day.append(pd.DataFrame([0, 0, 0, 0, 0]).T)
mean_week_day['day'].iloc[-1] = total_by_day['day'].iloc[-1]
mean_week_day['mean'].iloc[-1] = total_by_day['amountSpent'].iloc[-1]
mean_week_day.sort_values(by='day', axis=0, inplace=True)
mean_week_day.reset_index(inplace=True, drop=True)

plt.figure()
plt.plot(total_by_day['day'], total_by_day['amountSpent'], color='blue')
plt.plot(mean_week_day['day'], mean_week_day['mean'], color='red')

# Ok. It doesn't look bad at all. Can we do better? Let's explore that.

# %%
# Let's do some exploration before jumping in modelling. Taking a look at 
# some rows to explore any easy patterns. 
# This is a very important step. We'll try some plots and some
# exploratory data analysis here. 
# This is important because sometimes just looking at the statistics can
# be trick. That was shown by Anscombe in his famous quartet.
#
# Exploring 'emit', 'total' and 'versaoDocumento' we can see that these fields
# don't add any information. They're all the same or are just transactional
# data
#
# What about tables? This could be interesting to explore. Are people more
# likely to spend more depending on the table? Let's see.

df['tables'] = pd.Series([int(df['infAdic'].iloc[i]['infCpl'][5:]) for i
                          in range(0, df.shape[0])])
sns.lmplot('tables', 'amountSpent', df)

# Well, not that exciting. We really can't see any correlation here.
# Obviously it depends on the size of the dataset. Maybe one could cluster the
# analysis by groups of tables, or could try to find any correlation between
# days and tables used etc. There are plenty of possibilities here, but let's
# focus on what's more likely to help us: the products consumed.

# %%
# First of all, what are the products sold in that restaurant?

products = list(set([df['dets'].iloc[i][j]['prod']['xProd'] for i
                     in range(0, df.shape[0]) for j
                     in range(0, len(df['dets'].iloc[i]))]))

# Well, it looks like a japanese-bar restaurant.
# It would be interesting to see how much people would spent on lunch or dinner

sum_hour = pd.pivot_table(data=df, values='amountSpent', index='hour',
                               columns='weekday', aggfunc='mean')
sns.lmplot('hour', 'amountSpent', df)

# Apparently people tend to spend more on dinner in that restaurant.

# %%
# Ok, now we know better our problem, let's model the forecast using
# an ARIMA model

from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

# prepare data
X = total_by_day['amountSpent'].values
X = X.astype('float32')

history = [x for x in X]
predictions = list()

# The parameters of the model were found trying different combinations instead
# of doing a grid-search. I did this way because I wasn't very familiar
# time series, so I prefered to see the effect of changing the p, d and q
# manually.
model = ARIMA(history, order=(5,2,1))
model_fit = model.fit(disp=0)
forecast = model_fit.forecast(steps=6)[0]
for f in forecast:
    predictions.append(f)

# Let's artificially validate that prediction on the mean guess we've done.
# Here we'll be using the MAPE metric - Mean Absolute Percentage Error. 
# "It is a metric widely used in the field of time series forecasting, and
# refers to the average percentage of errors in the forecasts, disregarding
# the direction (above or below the true value).
# Source: http://mariofilho.com/create-simple-machine-learning-model-predict-time-series/

obs = mean_week_day['mean'][1:]


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape = mape(obs, predictions)
print('MAPE: %.3f' % mape)

# Let's see how we performed

pred = pd.DataFrame([list(range(6)), predictions]).T
pred.columns=['day', 'spent']
pred['day'] = pred['day'] + 25
pred = pred.append(pd.DataFrame([0, 0]).T)
pred['day'].iloc[-1] = total_by_day['day'].iloc[-1]
pred['spent'].iloc[-1] = total_by_day['amountSpent'].iloc[-1]
pred.sort_values(by='day', axis=0, inplace=True)
pred.reset_index(inplace=True, drop=True)

plt.figure()
plt.plot(total_by_day['day'], total_by_day['amountSpent'], color='blue')
plt.plot(mean_week_day['day'], mean_week_day['mean'], color='red')
plt.plot(pred['day'], pred['spent'], color='green')

# %%
# Conclusion: for a first try I think we did kinda well on that model. As
# suggested I didn't spend more than 5 hours, but I'd like to model the
# amount spent using the hours, for example, to see if with more data we could
# achieve a better result.
