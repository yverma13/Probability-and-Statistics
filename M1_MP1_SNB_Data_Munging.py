### DATASET ###

# The dataset chosen for this experiment is the Play Store dataset which is publicly available
# This dataset consists of 10841 records. Each record is made up of 13 fields

# Before we can derive any meaningful insights from the Play Store data, it is essential to pre-process 
# the data and make it suitable for further analysis. This pre-processing step forms a major part of data 
# wrangling (or data munging) and ensures better quality data. It consists of the transformation and mapping
# of data from a "raw" data form into another format so that it is more valuable for a variety of downstream
# purposes such as analytics. Data analysts typically spend a sizeable amount of time in the process of data
# wrangling, compared to the actual analysis of the data.

# After data munging is performed, several actionable insights can be derived from the Play Store apps data.
# Such insights could help to unlock the enormous potential to drive app-making businesses to success.

### IMPORT REQUIRED PACKAGES ###
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

### LOADING DATASET ###

playstore_data = pd.read_csv('googleplaystore.csv')
playstore_data.head()

''' Task 1: Data Cleaning '''

# Check whether there are any null values and figure out how you want to handle them?
# If there is any duplication of a record, how would you like to handle it?
# Are there any non-English apps? And how to filter them?
# In the size column, multiply 10,000,000 with entries having M and multiply by 10,000 if we have K in the cell

playstore_data.isna().sum()
playstore_data.dropna(inplace=True)

# Identify the duplicate apps
len(set(playstore_data['App'].values)), playstore_data.shape

# Remove the duplicate apps
playstore_data = playstore_data.drop_duplicates(['App'],keep='first')

# Check for any null values
playstore_data.isnull