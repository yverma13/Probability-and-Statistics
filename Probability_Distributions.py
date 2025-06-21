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
sns.displot(data_normal)
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
sns.displot(data_binom, kde=False)
df_b = pd.DataFrame({'x':data_binom})
df_b['P'] = df_b['x'].apply(lambda v: (math.factorial(n)/(math.factorial(n-v)*math.factorial(v)))*(p**v)*(q**(n-v)))
plt.plot(df_b['x'],10000*df_b['P'], '.')
plt.xlabel('Binomial Distribution')
plt.ylabel('Frequency')
plt.show()


### BERNOULLI DISTRIBUTION ###

# Consider a random experiment of tossing a biased coin (having the probability of getting a head as 0.6) once. 
# Plot the distribution associated with the event of getting heads in the given experiment if the process is repeated 10000 times.

p=0.6                      # probability of getting a head
q=1-p                      # probability of getting a tail

data_bern = bernoulli.rvs(p,size=10000)
sns.displot(data_bern, kde=False)
df_br = pd.DataFrame({'x': data_bern})
df_br['P'] = df_br['x'].apply(lambda v: (p**v)*(q**(1-v)))
plt.plot(df_br['x'],10000*df_br['P'],'.')
plt.xlabel('Bernoulli Distribution')
plt.ylabel('Frequency')
plt.show()


### GEOMETRIC DISTRIBUTION ###

# Plot the geometric distribution of size 10000 having the probability of success as 0.5

p = 0.5              # probability of success
q = 1-p              # probability of failure

data_geom = geometric(p,10000)
sns.displot(data_geom, kde=False)
df_g = pd.DataFrame({'x':data_geom})
df_g['P']=df_g['x'].apply(lambda v:p*q**(v-1))
plt.plot(df_g['x'], 10000*df_g['P'],'.')
plt.show()


### POISSON DISTRIBUTION ###

# Plot the poisson distribution having size 10000 and given rate parameter as 4

rate = 4

data_poisson = poisson.rvs(rate,size=10000)
sns.displot(data_poisson,bins=30,kde=False)
df_p = pd.DataFrame({'x':data_poisson})
df_p['P'] = df_p['x'].apply(lambda v: (rate**v)*(np.e**(-rate))/math.factorial(v))
plt.plot(df_p['x'],10000*df_p['P'],'.')
plt.xlabel('Poisson Distribution')
plt.ylabel('Frequency')
plt.show()


### UNIFORM DISTRIBUTION ###

# Plot the uniform distribution of size 10000 over the range {10,30}

n = 10000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc=start, scale=width)
sns.displot(data_uniform)
df_u = pd.DataFrame({'x':data_uniform})
df_u['P'] = df_u['x'].apply(lambda v: 1/(width))
plt.plot(df_u['x'],df_u['P'],'-')
plt.xlabel('Uniform Distribution ')
plt.ylabel('Frequency')
plt.show()


### CENTRAL LIMIT THEOREM ###

# Let's take a population that is exponentially distributed and check if CLT holds.

# Exponential distribution is a continuous distribution that is often used to model the expected time one needs 
# to wait before the occurrence of an event. The main parameter of exponential distribution is the 'rate' parameter λ, 
# such that both mean and standard deviation of the distribution are given by (1/λ).

# Assuming λ=0.25, the mean and standard deviation of the population can be calculated.

rate = 0.25 
mu = 1/rate 
sd = 1/rate 
print('Population mean:', mu)
print('Population standard deviation:', sd)

# Visualize an exponential distribution having size= 10000

data = np.random.exponential((1/rate),10000)
sns.displot(data)

# Now let's see how the sampling distribution looks for this population. 
# Consider two cases, i.e. with a small sample size (n= 2), and a large sample size (n=500).

''' Case 1: Draw 50 random samples from the population of size 2 each '''

# Drawing 50 random samples of size 2 from the exponentially distributed population
sample_size = 2
df2 = pd.DataFrame(index=['x1','x2'])

for i in range(1,51):
    exponential_sample = np.random.exponential(mu, sample_size)
    col = f'sample {i}'
    df2[col] = exponential_sample
df2

# For each of the 50 samples, the sample mean and its distribution plot is given as
df2_sample_means = df2.mean()
sns.displot(df2_sample_means)

''' Case 2: Repeat the above process with a much larger sample size (n=500) '''

# Drawing 50 random samples of size 500
sample_size = 500
df500 = pd.DataFrame()

for i in range(1,51):
    exponential_sample = np.random.exponential(mu, sample_size)
    col = f'sample {i}'
    df500[col] = exponential_sample

df500_sample_means = pd.DataFrame(df500.mean(),columns=['Sample means'])
sns.displot(df500_sample_means)

#The first 5 values from the 50 sample means
df500_sample_means.head()

# An estimate of the standard deviation of the sampling distribution can be obtained as:
np.std(df500_sample_means).values[0]

# The above value is very close to the value stated by the CLT, which is σ/√n:
sd/ np.sqrt(sample_size)

