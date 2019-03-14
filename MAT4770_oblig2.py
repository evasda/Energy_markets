import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
from scipy.optimize import curve_fit
import math as math
from datetime import datetime
from scipy.linalg import expm

""" Need to add a vector of time so that i can use it in the seasonality function."""


df = pd.read_excel('temp-data.xlsx', sheet_name='Sheet 1', header = 1, usecols = 'A:D')

"""Change date-format to datetime. """
df.Date = df.Date.apply(lambda x: str(x))
df.Date = df.Date.apply(lambda x: str(x[0:4] + '-' + x[4:6] + '-' + x[6:8]))
df.Date = pd.to_datetime(df['Date'], format = '%Y-%m-%d', exact=True)
df.set_index(df['Date'])

"""Plot time series. """
"""
df.plot('Date','Average', style='-b', linewidth = 0.5)
plt.grid('on')
plt.show()
"""

#t1 = [i%365 for i in range(0, len(df.Date))]
t = np.linspace(0, len(df.Average), len(df.Average), dtype=int)

"""Define seasonality function. """
def func(t, a0, a1, a2, a3):
	y = a0 + a1*t + a2*np.cos(2*np.pi*(t - a3)/365)
	return y

"""Get the optimized coefficients by least square and store as opt_coeff."""
results = curve_fit(func, t, np.asarray(df.Average))

opt_coeff = results[0]
a0 = results[0][0]
a1 = results[0][1]
a2 = results[0][2]
a3 = results[0][3]
print(opt_coeff)



"""Create list of seasonal function values and convert to column in dataframe."""
season = []
for i in range(len(t)):
	season.append(func(t[i], a0, a1, a2, a3))
df['Season'] = pd.Series(season)

"""Plot seasonal curve with plot of temperatures."""
df.plot(x ='Date', y = ['Average', 'Season'], style=['-b', '-r'], linewidth = 0.8)
plt.grid('on')
plt.show()

"""Remove seasonal effect and store as column in dateframe."""
x = df.Average - df.Season
df['x'] = pd.Series(x)

"""Plot autocorrelation and partial autocorrelation."""
#autocorrelation_plot(x)
#plot_pacf(x, lags=50)
#plt.show()
# We clearly see that lag 3 is the right choice.


#sm.OLS(x, lag_func(x))
model = AR(x)
model_fit = model.fit(maxlag = 40, ic = 'aic', trend = 'nc')
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
print('Residuals: %s' % model_fit.sigma2)

"""Find CAR(3) coefficients from AR(3) coefficients."""
alpha1 = 3 - model_fit.params[0]
alpha2 = 2 * alpha1- model_fit.params[1] - 3
alpha3 = alpha2 - alpha1 - model_fit.params[2] + 1
#print(alpha1, alpha2, alpha3)


"""Create column with year, month and day."""
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day
aug =  df['Month']==8
#print(df[aug])
df_aug = df[aug]

"""CDD for the month of August for every year."""
c = 18.0
CDD = []
for i in range(1995, 2005):
	CDD_1 = 0
	temp = list(df_aug[df_aug.Year==i].Average.values)
	for j in temp:
		m = max(j - c, 0)
		CDD_1 += m
	CDD.append(CDD_1)

print(CDD)
def average():
	print(len(CDD))
	a = sum(CDD)/len(CDD)
	return a

kappa = average()
print('Estimated kappa: %f' % kappa)

"""
Need to compute the expected value of CDD index for august, with t=july 1. Can assume that we are at seasonal mean 1.july.
We therefore need a model that starts at 1.july where T=Season, i.e. X=0 for all lags. But then X will be zero forever, so clearly 
this is not correct. 
"""

#model_fit.predict(np.asarray(alpha1, alpha2, alpha3), start = datetime(1995, 7, 1) , end =  datetime(1995, 8, 31))

temp_pred = []

def func1():
	X_pred = []
	norm = np.random.standard_normal()
	X_pred.append(0)
	X_pred.append(model_fit.sigma2 * norm)
	norm = np.random.standard_normal()
	X_pred.append((3 - alpha1)*X_pred[1] + model_fit.sigma2 * norm)
	print(3 - alpha1)
	print(2*alpha1 - alpha2 - 3)
	print(alpha2 + 1 - alpha1 - alpha3)
	for j in range(3, 62):
		norm = np.random.standard_normal()
		X_pred.append((3 - alpha1)*X_pred[j-1] + (2*alpha1 - alpha2 - 3)*X_pred[j-2] + (alpha2 + 1 - alpha1 - alpha3) * X_pred[j-3] + model_fit.sigma2 * norm)
	#print(X_pred)
	return(X_pred)

S = df.Season[182:244] 		# Seasonal values for the months of july and august.

N = 62						# Number of days to simulate
A = np.array([[0, 1, 0], [0, 0, 1], [-alpha3, -alpha2, -alpha1]])
s = np.linspace(0, N, N, dtype=int)


def X_car3(si, i):
	temp = np.zeros((3,3))
	for j in s[0:i]:
		dbm = np.random.standard_normal(1)
		temp += (expm(A*(si-j)) * np.identity(3) * sigma2 * dbm)
	return(temp)


sim = 500					# Number of simulations.
sigma2 = model_fit.sigma2
temp_arr = np.zeros((N, sim))
err_arr = np.zeros((N, sim))
total = np.zeros((N, sim))
counter = np.zeros(sim)
for j in range(0, sim):
	norm = np.random.standard_normal(N)
	err_arr[:,j] = sigma2 * norm
	temp_arr[1,:] = sigma2 * err_arr[0,]
	temp_arr[2,] = (3 - alpha1)*temp_arr[1,] + sigma2 * err_arr[1,]
	#for i in range(1, len(s)):											# removed temporarily (car)
	for i in range(3, N):
		temp_arr[i,] = (3 - alpha1)*temp_arr[i-1,:] + (2*alpha1 - alpha2 - 3)*temp_arr[i-2,:] + (alpha2 + 1 - alpha1 - alpha3) * temp_arr[i-3,:] + sigma2 * err_arr[i-3,:]
		#temp_arr[i,] = X_car3(s[i],i)[0,0]								# removed temporarily (car)
	total[:,j] = temp_arr[:,j] + df.Season[182:244]

	counter[j] = 0

	for i in range(31, N):
		counter[i] = 0
		if total[i,j] <= kappa:
			counter[j] += 1
print(counter)

expected_cdd = sum(counter)/sim
print('Expected value CDD less than kappa: %f' % expected_cdd)

#print(pred_season())
x_ = np.asarray(func1())
xx = x_.flatten()
pred_aug = xx + S
#print(pred_aug)
plt.plot(pred_aug)
plt.show()


X = []
X.append(0)
e = np.transpose(np.array([1, 0, 0]))
for i in range(1, len(s)):
	X.append(X_car3(s[i],i)[0,0])

Y = []
Y.append(X + func(s[i], a0, a1, a2, a3))

print(X)
plt.plot(X + S)
plt.show()

#print(X)


df = pd.read_excel('mandatory-forwardprices.xlsx', sheet_name='Sheet1')
df.columns = ['date', 'price']
print(df.tail())
df.plot('date','price', style='r')
plt.grid('on')
plt.show()
#print(data.loc['Forward price (Dollars)'])
#date = np.asarray(data.DATE)
#price = np.asarray(data.forward price (Dollars))
#print(list(data.columns))


"""Estimate the drift mu and the volatility sigma from the data set. """

df['logreturns'] = np.log(df.price.values)
df['dt'] = df.price.diff(periods= 1)

N = 1691
alpha = np.sum(df.logreturns)/(N-1)
print(alpha)

sigma_2 = 1/(N-2) * np.sum(np.square(df.logreturns.values - alpha))
mu = alpha + 1/2*sigma_2

d = 0.1
startvalue = 31600
sim = 10

for i in range(0, sim):
	F = np.zeros((N, sim))
	F[0, sim] = 0
	XX[0] 
	for j in range(0, 30):
		ss = np.linspace(0, 61-j, 61-j)
		np.random.standard_normal(sim)
		F[j,] = F[0] * np.exp(mu + sigma^2/2 * tt + sigma * bm)
		#xxx = X_car(s[i], i)