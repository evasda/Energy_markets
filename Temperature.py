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

class Temperature:
	def __init__(self, n, dt):
		self.N = n 				# Number of days to simulate.
		self.t = np.linspace(0, n, dt)  # Range of values to simulate over. Default is days of simulation.

	def seasonality(self, t_av, a0, a1, a2, a3):
		"""Define seasonality function. """
		self.y = a0 + a1*t_av + a2*np.cos(2*np.pi*(t_av - a3)/365)
		return self.y

	def fit_season(self, average_data):
		"""Get the optimized coefficients by least square and store as opt_coeff."""
		self.t_av = np.linspace(0, len(df.Average), len(df.Average), dtype=int)
		results = curve_fit(self.seasonality, self.t_av, np.asarray(average_data))
		opt_coeff = results[0]
		self.a0 = results[0][0]
		self.a1 = results[0][1]
		self.a2 = results[0][2]
		self.a3 = results[0][3]
		print('Estimated coefficients for seasonal function: %s' % opt_coeff)
		return(self.a0, self.a1, self.a2, self.a3)

	def est_CAR3(self, x):
		"""
    	Estimates AR-model. Input is the deseasonalized temperature computed from data.
    	Prints lag, coefficients and residuals.
    	"""
		model = AR(x)
		model_fit = model.fit(maxlag = 40, ic = 'aic', trend = 'nc')
		self.volatility = model_fit.sigma2
		print('Ar lag: %s' % model_fit.k_ar)
		print('AR coefficients: %s' % model_fit.params)
		print('AR residuals: %s' % model_fit.sigma2)

		"""Find CAR(3) coefficients from AR(3) coefficients."""
		self.alpha1 = 3 - model_fit.params[0]
		self.alpha2 = 2 * self.alpha1- model_fit.params[1] - 3
		self.alpha3 = self.alpha2 - self.alpha1 - model_fit.params[2] + 1
		print('Estimated CAR(p)-coefficients: %s' % [self.alpha1, self.alpha2, self.alpha3])
		return(self.alpha1, self.alpha2, self.alpha3, self.volatility)

	def CDD_estimate(self, dataframe, year_start=0, year_end=1):
		"""CDD for the month of August for every year."""
		month = dataframe
		c = 18.0
		self.CDD = []
		for i in range(year_start, year_end+1):
			CDD_1 = 0
			temp = list(month[month.Year==i].Average.values)
			for j in temp:
				m = max(j - c, 0)
				CDD_1 += m
			self.CDD.append(CDD_1)
		print('Estimated CDD: %s' % self.CDD)
		return(self.CDD)

	def average(self):
		self.kappa = sum(self.CDD)/len(self.CDD)
		print('Estimated kappa: %f' % self.kappa)
		return(self.kappa)

	def temp_estimate_AR3(self, d_start, d_end, season_df):
		sigma2 = self.volatility
		alpha1 = self.alpha1
		alpha2 = self.alpha2
		alpha3 = self.alpha3
		A = np.array([[0, 1, 0], [0, 0, 1], [-alpha3, -alpha2, -alpha1]])
		N = self.N
		temp_arr = np.zeros(N)
		err_arr = np.zeros(N)
		total = np.zeros(N)
		norm = np.random.standard_normal(N)
		err_arr[0:N] = sigma2 * norm
		temp_arr[1] = sigma2 * err_arr[1]
		temp_arr[2] = (3 - alpha1)*temp_arr[1] + sigma2 * err_arr[1]
		for i in range(3, N):
			temp_arr[i] = (3 - alpha1)*temp_arr[i-1] + (2*alpha1 - alpha2 - 3)*temp_arr[i-2] + (alpha2 + 1 - alpha1 - alpha3) * temp_arr[i-3] + sigma2 * err_arr[i-3]
		total[0:N] = temp_arr[0:N] + season_df[d_start:d_end]
		return(total)

	def temp_estimate_CAR3(self, ss, i):
		sigma2 = self.volatility
		temp = np.zeros((3,3))
		for j in ss[0:i]:
			dbm = np.random.standard_normal(1)
			temp += (expm(A*(ss-j)) * np.identity(3) * sigma2 * dbm)
		return(temp)

	def MCMC_temperature(self, sim, season_df):
		#N = number_days
		sigma2 = self.volatility
		temp_arr = np.zeros((N, sim))
		err_arr = np.zeros((N, sim))
		total = np.zeros((N, sim))
		counter = np.zeros(sim)
		for j in range(0, sim):
			total[:,j] = self.temp_estimate_AR3(d_start = 182, d_end = 244, season_df = season_df)

			counter[j] = 0
			if np.sum(total[31:N,j]) < self.kappa:
				counter[j] += 1
		print('Counter value: %s' % counter)

		expected_cdd = sum(counter)/sim
		print('Expected CDD: %s' % expected_cdd)

	def estimate_futures(t, tau1, tau2, sim):
		startvalue = 31600
		for i in range(0, sim):
			F = np.zeros((N, sim))
			F[0, sim] = 0
			XX[0] 
			for j in range(0, 30):
				ss = np.linspace(0, 61-j, 61-j)
				np.random.standard_normal(sim)
				F[j,] = F[0] * np.exp(mu + sigma^2/2 * tt + sigma * bm)
				XX[j,] = X_car(s[i], i)



if __name__ == '__main__':
	df = pd.read_excel('temp-data.xlsx', sheet_name='Sheet 1', header = 1, usecols = 'A:D')

	"""Change date-format to datetime. """
	df.Date = df.Date.apply(lambda x: str(x))
	df.Date = df.Date.apply(lambda x: str(x[0:4] + '-' + x[4:6] + '-' + x[6:8]))
	df.Date = pd.to_datetime(df['Date'], format = '%Y-%m-%d', exact=True)
	df.set_index(df['Date'])

	"""Create column with year, month and day."""
	df['Year'] = pd.DatetimeIndex(df['Date']).year
	df['Month'] = pd.DatetimeIndex(df['Date']).month
	df['Day'] = pd.DatetimeIndex(df['Date']).day
	aug =  df['Month']==8
	df_aug = df[aug]

	#"""Plot time series. """
	#df.plot('Date','Average', style='-b', linewidth = 0.5)
	#plt.grid('on')
	#plt.show()

	n= 62
	dt = 62
	t = np.linspace(0, n, dt)  # Range of values to simulate over. Default is days of simulation.
	tempt = Temperature(n, dt)
	[a0, a1, a2, a3] = tempt.fit_season(df.Average)
	#tempt.fit_season()

	"""Create list of seasonal function values and convert to column in dataframe."""
	season = []
	t_av = np.linspace(0, len(df.Average), len(df.Average), dtype=int)
	for i in range(len(t_av)):
		season.append(tempt.seasonality(t_av[i], a0, a1, a2, a3))
	df['Season'] = pd.Series(season)

	"""Plot seasonal curve with plot of temperatures."""
	df.plot(x ='Date', y = ['Average', 'Season'], style=['-b', '-r'], linewidth = 0.8)
	plt.grid()
	plt.show()

	"""Remove seasonal effect and store as column in dateframe."""
	x = df.Average - df.Season
	df['x'] = pd.Series(x)

	tempt.est_CAR3(df.x.values)
	tempt.CDD_estimate(dataframe = df_aug, year_start = 1995, year_end = 2004)		
	tempt.average()


	S = df.Season[182:244] 		# Seasonal values for the months of july and august.

	N = 62						# Number of days to simulate
	s = np.linspace(0, N, N, dtype=int)

	tempt.MCMC_temperature(sim = 500, season_df = df.Season)

	#print(pred_season())
	x_ = np.asarray(func1())
	xx = x_.flatten()
	pred_aug = xx + S
	#print(pred_aug)
	plt.plot(pred_aug)
	plt.show()