import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy.stats import binom, nbinom, bernoulli, poisson, norm, uniform
from numpy.random import geometric
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings("ignore")

### NORMAL DISTRIBUTION ###

# Plot a normal distribution of size 10000 having mean as 0 and standard deviation as 1
# generate random numbers from N(0,1)
# .rvs provides random samples; scale = std. deviation, loc = mean, size = no. of samples

mean = 0
std = 1

data_normal = norm.rvs(size=10000,loc=mean,scale=std)
sns.distplot(data_normal)
df_n = pd.DataFrame({'x':data_normal})
df_n['P'] = df_n['x'].apply(lambda v: np.exp(-(((v - mean)/std)**2)/2)/(std*np.sqrt(2*np.pi)))
df_n = df_n.sort_values(by='x')
plt.plot(df_n['x'],df_n['P'],'-')
plt.xlabel('Normal Distribution')
plt.ylabel('Density')
plt.show()


### BINOMIAL DISTRIBUTION ###

# Consider an event where a fair coin is tossed 10 times and the total number of heads is recorded. 
# Plot the distribution for the event.

# Generate data
p = 1/2             # probability of getting a head
q = 1-p             # probability of getting a tail
n = 10              # total number of trials

data_binom = binom.rvs(n,p,size=10000)
sns.distplot(data_binom, kde=False)
df_b = pd.DataFrame({'x':data_binom})
df_b['P'] = df_b['x'].apply(lambda v: (math.factorial(n)/(math.factorial(n-v)*math.factorial(v)))*(p**v)*(q**(n-v)))
plt.plot(df_b['x'],10000*df_b['P'], '.')
plt.xlabel('Binomial Distribution')
plt.ylabel('Frequency')
plt.show()