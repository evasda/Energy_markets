from Temperature import Temperature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(Temperature)


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


tempt = Temperature(n=62, dt = 62)
tempt.seasonality()
#tempt.fit_season()

"""Create list of seasonal function values and convert to column in dataframe."""
season = []
for i in range(len(t)):
	season.append(func(t[i], a0, a1, a2, a3))
df['Season'] = pd.Series(tempt.seasonality())

"""Plot seasonal curve with plot of temperatures."""
df.plot(x ='Date', y = ['Average', 'Season'], style=['-b', '-r'], linewidth = 0.8)
plt.grid('on')
plt.show()

"""Remove seasonal effect and store as column in dateframe."""
x = df.Average - df.Season
df['x'] = pd.Series(x)
