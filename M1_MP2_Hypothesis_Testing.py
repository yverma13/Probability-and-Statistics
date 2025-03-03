### Dataset ###

# The dataset chosen for this experiment is the ab_data.csv which is publicly available on Kaggle
# This dataset consists of 2,94,478 records. Each record is made up of 5 fields.

# For example, Each record consists of 'user_id', 'timestamp', 'group', 'landing_page' and 'converted'.
# user_id: A unique identifier assigned to each user (i.e., a visitor to the company's webpage) participating in the experiment.
# timestamp: The timestamp indicating the time at which the user interacted with the webpage or was exposed to the experimental condition.
# group: The group to which the user was assigned, typically denoted as either 'treatment' or 'control'. This field helps categorize users into different experimental conditions.
# landing_page: Specifies the type of landing page or webpage variant that the user was directed to upon interaction. It distinguishes between different versions of the webpage used in the experiment.
# converted: A binary indicator representing whether the user performed the desired action or conversion after interacting with the webpage. It typically indicates whether the user made a purchase, signed up for a service, or completed any other desired action.

### Problem Statement ###

# The biggest e-commerce company called FaceZonGoogAppFlix approached a data science consulting firm as a new client!
# They have a potential new webpage designed with the intention to increase their current conversion rates of 12% by 0.35% or more. 
# With such an ambiguous task, they have full trust in the data science consulting firm to give them a recommendation whether to implement the new web page or keep the old webpage. 
# Unfortunately they haven't built up a data science capability in their company, but they've used an external software called 'A/B Tester' for 23 days and then come back to the data science consulting firm with a dataset (ab_data.csv). 
# Under this requirement scenario, what should the data science consulting firm do?

### IMPORT REQUIRED PACKAGES ###
import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import math as mt
import itertools
import random
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from scipy.stats import norm

### Load the dataset ###
data = pd.read_csv('ab_data.csv')
df = data.copy()
df.head()

df['group'].value_counts()

# Finding the number of rows in the dataset
print('The number of rows in the dataset:',df.shape[0])

''' Task 1: Data Cleaning '''

# Check the number of unique users in the dataset
# Check the proportion of users converted. Hint: query(), count()
# Estimate how many times the new_page and treatment don't line up. Also estimate how many times the old_page and control do not match.
# Display the total no. of non-line up pages
# Check if any of the rows have missing values?

# The number of unique users in the dataset
df.user_id.nunique()

# The proportion of users converted
df.query('converted == 1')['converted'].count() / df.shape[0]

# Identify treatment does not match with new_page
N1 = df.query('group == "treatment" and landing_page != "new_page"').count()[0]
N1

# Identify control does not match with old_page
N2 = df.query('group != "treatment" and landing_page == "new_page"').count()[0]
N2

# Total no. of non-line up
N = N1 + N2
N

# Check for any missing values
df.isnull().sum().sum()

# Check datatype of each column
df.dtypes

''' Task 2: Identify the not aligned rows '''