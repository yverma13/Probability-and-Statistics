# Importing required packages

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm           # A normal continuous random variable
from numpy.random import seed          # set the randomness
from scipy import stats                # statistical operations
import warnings
warnings.simplefilter("ignore")

# Example 1: Let's create some dummy age data for the population of voters in the entire country and a sample 
# of voters in North_Carolina and test whether the average age of voters in North_Carolina differs from the 
# entire country population.

np.random.seed(6)

# Generate Population ages data, see this https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
# .rvs provides random samples; scale = std. deviation, loc = mean, size = no. of samples

population_ages1 = stats.norm.rvs(scale=18, loc=45, size=150000)
population_ages2 = stats.norm.rvs(scale=18, loc=20, size=100000)
population_ages = np.concatenate((population_ages1,population_ages2))

# Generate North Carolina sample ages data

North_Carolina_ages1 = stats.norm.rvs(scale=18, loc=40, size=30)
North_Carolina_ages2 = stats.norm.rvs(scale=18, loc=20, size=25)
North_Carolina_ages = np.concatenate((North_Carolina_ages1, North_Carolina_ages2))

print("Mean age of population of voters in the entire country:",population_ages.mean())
print("Mean age of North Carolina population:",North_Carolina_ages.mean())

# Perform Z-test for mean
one_sample_ztest = sm.stats.ztest(x1= North_Carolina_ages, x2=None, value= population_ages.mean(), alternative='smaller')
print('p-value: ', one_sample_ztest[1])

# Example 2: Let's create a sample of voters in South_Carolina and test whether the average age of voters in 
# North_Carolina differs from the average age of voters in South_Carolina.

np.random.seed(12)

# Generate South Carolina sample ages data
South_Carolina_ages1 = stats.norm.rvs(scale=15, loc=33, size=30)
South_Carolina_ages2 = stats.norm.rvs(scale=15, loc=20, size=25)
South_Carolina_ages = np.concatenate((South_Carolina_ages1, South_Carolina_ages2))

print('Mean age of South Carolina population:', South_Carolina_ages.mean())

# Perform Two sample Z-test for Mean
two_sample_ztest = sm.stats.ztest(x1=South_Carolina_ages1, x2=South_Carolina_ages2, value=0, alternative='two-sided')
print('test statistic(z-score):',two_sample_ztest[0])
print('p-value:',two_sample_ztest[1])

# Example 3: Consider the weights of a group of patients before and after an exercise program.
# Observation 1: The weight of a group of patients was evaluated at baseline.
# Observation 2: This same group of patients was evaluated after an 8-week exercise program.
# Variable of interest: Body weight.

np.random.seed(11)

# Generate Weights of population before exercise program
before = stats.norm.rvs(scale=10, loc=50, size=100)

# Generate Weights of population after exercise program
after = before + stats.norm.rvs(scale=5, loc=-3.5, size=100)

# Create dataframe 
weight_df = pd.DataFrame({"weight_before":before,"weight_after":after,"weight_change":after-before})

weight_df.head()

weight_df.describe()

# Perform paired Z-test
pair_ztest = sm.stats.ztest(x1=before,x2=after,value=0,alternative='two-sided')
print("test-statistic:",pair_ztest[0])
print("p-value:",pair_ztest[1])

# Example 4: Consider the Professional Salary Survey Results dataset. At  α  = 0.05, test whether the salary 
# means for both male and female employees for the year 2019 who belong to the United States, are equal.

# The corresponding null hypothesis and alternate hypothesis are as follows:
# H0 : both salary means( male and female) are equal
# HA : both means are not equal

import pandas as pd

# Read data
raw_data = pd.read_csv('2019_Data_Professional_Salary_Survey_Responses.csv', header = 3)
raw_data.head()

# Shape of data
raw_data.shape

len(raw_data)

# Check for missing values
raw_data.isna().sum()

# Check for missing values
raw_data.isna().sum()/len(raw_data)

# Remove missing values
raw_data.dropna(inplace=True)
raw_data.reset_index(drop=True, inplace=True)
raw_data.isna().sum()

# Data type of SalaryUSD column
raw_data['SalaryUSD'].dtype

# Unique values of SalaryUSD column
raw_data['SalaryUSD'].unique()

# Processing Salary column
def process_salary(salary):
    sep = '.'
    salary = str(salary)
    # Replace characters and take the cents out of our data
    salary = salary.replace(" ","").replace("$","").replace(",","").split(sep)[0]

    return float(salary)

# Replace spaces(“ “) in columns name with underscore (“_”)
raw_data.columns = raw_data.columns.str.replace(" ","_")

# Apply process_salary function
raw_data['SalaryUSD'] = raw_data['SalaryUSD'].apply(process_salary)
raw_data['SalaryUSD'].head()

# Filter dataframe by year
df = raw_data[raw_data.Survey_Year == 2019]
df.head()

# Filter dataframe by country
US_2019 = df.loc[df.Country == 'United States', :]
US_2019.head(3)

# Number of employees by their genders
US_2019.Gender.value_counts()

# Filter salary by gender
Male_salary = US_2019[US_2019.Gender == 'Male']['SalaryUSD']
Female_salary = US_2019[US_2019.Gender == 'Female']['SalaryUSD']

print("Mean salary for male employees: ", Male_salary.mean())
print("Mean salary for female employees: ", Female_salary.mean())

# Perform t-test on data
ttest = stats.ttest_ind(a = Female_salary, b = Male_salary, equal_var= False)
ttest