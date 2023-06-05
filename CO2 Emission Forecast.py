# I use a public dataset of monthly carbon dioxide emissions from electricity generation available at the Energy Information Administration and Jason McNeill. The dataset includes CO2 emissions from each energy resource starting January 1973 to July 2016 for reference click

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import statsmodels
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import SARIMAX

rcParams['figure.figsize'] = 20, 16

# Ignore warning messages when parsing CSV file
warnings.filterwarnings("ignore")

# Load CSV file into pandas dataframe
df = pd.read_csv("../input/MER_T12_06.csv")

# Display the first 5 rows of the dataframe and its metadata
df.head()
df.info()


# Parses specified column(s) into a pandas datetime object
def dateparse(x):
    return pd.to_datetime(x, format='%Y%m', errors='coerce')


# Parses first column of CSV file (YYYYMM) into datetime objects and makes it the index column
df = pd.read_csv("../input/MER_T12_06.csv", parse_dates=['YYYYMM'], index_col='YYYYMM', date_parser=dateparse)

# Displays the first 15 rows of the dataframe
df.head()

# Displays the last 15 rows of the dataframe
df.tail(15)


# Displays the value count of Column_Order
df['Column_Order'].value_counts()


# Extract only non-null datetime index values from the dataframe
ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce').notnull().values).notnull()].copy(deep=True)

# Coerce non-numeric value in 'Value' column to numeric and filter out any resulting NaN values
ts['Value'] = pd.to_numeric(ts['Value'], errors='coerce')

# Displays the head(15) rows of the filtered dataframe
ts.head(15)

# Displays the data type (dtypes) of the filtered dataframe
ts.dtypes


# Displays the metadata of the filtered dataframe
ts.info()


# Drops rows with NaN values
ts.dropna(inplace=True)

# Groups the dataframe by 'Description' column and performs plotting
Energy_sources = t




cols = ['Geothermal Energy', 'Non-Biomass Waste', 'Petroleum Coke', 'Distillate Fuel ',
        'Residual Fuel Oil', 'Petroleum', 'Natural Gas', 'Coal', 'Total Emissions']
plt.figure(figsize=(16, 9))
x_tick = np.arange(len(cols))
plt.bar(x_tick, CO2_per_source, align='center', alpha=0.5)
plt.suptitle("CO2 Emissions by Electric Power Sector", fontsize=25)
plt.xticks(x_tick, x_label, rotation=70, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Carbon Emissions in MMT', fontsize=20)
plt.show()

ts = pd.read_csv("../input/MER_T12_06.csv", parse_dates=['YYYYMM'], index_col='YYYYMM', date_parser=lambda x: pd.to_datetime(x))
Emissions = ts.iloc[:, 1:]
Emissions = Emissions.groupby(['Description', pd.TimeGrouper('M')])[
    'Value'].sum().unstack(level=0)
mte = Emission['Natural Gas Electric Power Sector CO2 Emissions']
mte.head()
mte.tail()

plt.plot(mte)

def TestStationaryPlot(ts):
    rol_mean = ts.rolling(window=12, center=False).mean()
    rol_std = ts.rolling(window=12, center=False).std()
    plt.plot(ts, color='blue', label='Original Data')
    plt.plot(rol_mean, color='red', label='Rolling Mean')
  



mte_first_difference = mte - mte.shift(1)
TestStationaryPlot(mte_first_difference.dropna(inplace=False))


TestStationaryAdfuller(mte_first_difference.dropna(inplace=False))


mte_seasonal_difference = mte - mte.shift(12)
TestStationaryPlot(mte_seasonal_difference.dropna(inplace=False))
TestStationaryAdfuller(mte_seasonal_difference.dropna(inplace=False))


mte_seasonal_first_difference = mte_first_difference - \
    mte_first_difference.shift(12)
TestStationaryPlot(mte_seasonal_first_difference.dropna(inplace=False))


TestStationaryAdfuller(mte_seasonal_first_difference.dropna(inplace=False))


decomposition = seasonal_decompose(mte)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(mte, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
mte_decompose = residual
mte_decompose.dropna(inplace=True)
TestStationaryPlot(mte_decompose)
TestStationaryAdfuller(mte_decompose)
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(
    mte_seasonal_first_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(
    mte_seasonal_first_difference.iloc[13:], lags=40, ax=ax2)

# Define the range of seasons for ARIMA
p = range(0, 2)
d = range(0, 1)
q = range(0, 1)

pdq = list(itertools.product(p, d, q))

pdq_x_QDQs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of Seasonal ARIMA parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], pdq_x_QDQs[1]))
print('SARIMAX: {} x {}'.format(pdq[2], pdq_x_QDQs[2]))

print(pdq)
print(pdq_x_QDQs)

for param in pdq:
    for seasonal_param in pdq_x_QDQs:
        try:
            mod = sm.tsa.statespace.SARIMAX(mte,
                                            order=param,
                                            seasonal_order=seasonal_param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            a = []
            b = []
            c = []
            wf = pd.DataFrame()
            warnings.filterwarnings("ignore")  # specify to ignore warning messages
    
            for param in pdq:
                for param_seasonal in pdq_x_QDQs:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(mte,
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                        results = mod.fit()
                        a.append(param)
                        b.append(param_seasonal)
                        c.append(results.aic)
                    except:
                        continue




# Load the dataset
pd.read_csv('mte.csv', index_col='Date')

# Split the data into training and testing sets
X = mte.drop(index=pd.to_datetime('2013-01-31'), axis=1)
y = mte['2013-01-31']:

# Train the ARIMA model on the training set
model = SARIMAX(y, order=(-2., 1, 0), seasonal_order=(0, 1, 1), enforce_stationarity=True, enforce_invertibility=True)
results = model.fit(disp=False)

# Get the predicted values and confidence intervals for the test set
pred_dynamic = results.get_prediction(start=pd.to_datetime(
    '2013-01-31'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Get the forecasted values for the next 120 months
forecast_df = results.get_forecast(steps=120)

# Create the plot
ax = mte['1973':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(ax=ax, label='Dynamic Forecast')

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 1],
                pred_dynamic_ci.iloc[:, 0], color='r', alpha=.3)

ax.fill_betweenx(ax.get_ylim(),
                 pd.to_datetime('2013-01-31'),
                 x[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('CO2 Emissions')

plt
