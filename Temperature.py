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

class DoublePendulum:
    def __init__():


    def __call__(self, t, y):


    """Define seasonality function. """
	def seasonality(t, a0, a1, a2, a3):
		y = a0 + a1*t + a2*np.cos(2*np.pi*(t - a3)/365)
		return y

	"""Get the optimized coefficients by least square and store as opt_coeff."""
	def fit_season(t, average_data):
		results = curve_fit(self.seasonality, t, np.asarray(average_data))
		opt_coeff = results[0]
		a0 = results[0][0]
		a1 = results[0][1]
		a2 = results[0][2]
		a3 = results[0][3]
		print(print('Estimated coefficients for seasonal function: %s' % opt_coeff))

	def AR_estimation(x):
		model = AR(x)
		model_fit = model.fit(maxlag = 40, ic = 'aic', trend = 'nc')
		self.volatility = model_fit.sigma2
		print('Lag: %s' % model_fit.k_ar)
		print('Coefficients: %s' % model_fit.params)
		print('Residuals: %s' % model_fit.sigma2)

		"""Find CAR(3) coefficients from AR(3) coefficients."""
		self.alpha1 = 3 - model_fit.params[0]
		self.alpha2 = 2 * self.alpha1- model_fit.params[1] - 3
		self.alpha3 = self.alpha2 - self.alpha1 - model_fit.params[2] + 1
		print('Estimated CAR(p)-coefficients: %s' % [self.alpha1, self.alpha2, self.alpha3])
		return(self.alpha1, self.alpha2, self.alpha3, self.volatility)

	"""CDD for the month of August for every year."""
	def CDD_estimate(year_start=0, year_end=2, month):
		c = 18.0
		CDD = []
		for i in range(1995, 2005):
			CDD_1 = 0
			temp = list(df_aug[df_aug.Year==i].Average.values)
			for j in temp:
			m = max(j - c, 0)
			CDD_1 += m
			CDD.append(CDD_1)

			print('Estimated CDD: %s' % CDD)

	def average():
		a = sum(CDD)/len(CDD)
		return a

	def temp_estimate_AR3(N, sigma2, alpha1, alpha2, alpha3, d_start, d_end):
		temp_arr = np.zeros(N)
		err_arr = np.zeros(N-1)
		total = np.zeros(N)
		norm = np.random.standard_normal(N)
		err_arr[0,N-1] = sigma2 * norm
		temp_arr[1] = sigma2 * err_arr[0]
		temp_arr[2] = (3 - alpha1)*temp_arr[1] + sigma2 * err_arr[1]
		for i in range(3, N):
			temp_arr[i] = (3 - alpha1)*temp_arr[i-1] + (2*alpha1 - alpha2 - 3)*temp_arr[i-2] + (alpha2 + 1 - alpha1 - alpha3) * temp_arr[i-3] + sigma2 * err_arr[i-3]
		total[0:N] = temp_arr[0:N] + df.Season[d_start:d_end]
		return(total)

	def temp_estimate_CAR3(si, i):
		temp = np.zeros((3,3))
		for j in si[0:i]:
			dbm = np.random.standard_normal(1)
			temp += (expm(A*(si-j)) * np.identity(3) * sigma2 * dbm)
		return(temp)

	def MCMC_temperature(sim, number_days, season_df):
		N = number_days
		sigma2 = self.volatility
		temp_arr = np.zeros((N, sim))
		err_arr = np.zeros((N, sim))
		total = np.zeros((N, sim))
		counter = np.zeros(sim)
		for j in range(0, sim):
			total[:,j] = temp_arr[:,j] + df.Season[182:244]

		counter[j] = 0

		for i in range(31, N):
			counter[i] = 0
			if total[i,j] <= kappa:
				counter[j] += 1
			print('Counter value: %s' % CDD)

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
				F[j,] = F[0] * np.exp(mu + sigma^2/2 * tt + sigma * bm
				xxx = X_car(s[i], i)