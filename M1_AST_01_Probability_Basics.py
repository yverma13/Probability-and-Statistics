# Importing required packages
import numpy as np
import pandas as pd
import scipy                       
import matplotlib.pyplot as plt     
import seaborn as sns               
from scipy import integrate         
sns.set_style('whitegrid')

# Load the dataset
df_ = pd.read_csv('garments_worker_productivity.csv')

#Explore and preprocess dataset
df_.head()

# Consider only five features from dataset
df = df_[['date','quarter','department','day','team']]

# Consider records where 'day' is Monday, Thursday or Saturday
df_day = df[df['day'].isin(['Monday','Thursday','Saturday'])]

# Consider records where 'team' number is 1, 2 or 3
df_day_team = df_day[df_day['team'].isin([1,2,3])]

# Consider records where 'quarter' is 'Quarter1' or 'Quarter2'
df_day_team_quarter = df_day_team[df_day_team['quarter'].isin(['Quarter1','Quarter2'])]

# Reset the index and store dataset to 'df'
df = df_day_team_quarter.reset_index(drop=True)

# Check for unique values in department column
df['department'].unique()

# Remove extra space from 'finishing ' department column
df['department'] = df['department'].apply(lambda x: x.replace(' ',''))

# Change department from 'sweing' to 'sewing'
for i in range(len(df)):
    if df.loc[i,'department']=='sweing':
        df.loc[i,'department']='sewing'

# Check for unique values in department column
df['department'].unique()

# Display few rows of processed dataset
df.sample(5)

print('Dataset shape before processing: ', df_.shape)
print('Dataset shape after processing: ', df.shape)

# Select a record from the above given dataset
i1 = np.random.randint(df.shape[0]-1)
record = df.iloc[i1:i1+1, :]             
record

# Calculate the length of sample space for a random experiment of selecting a record from the above given dataset
len(df.index)

# Getting a finishing department record is an event related to the experiment of selecting a record from the whole dataset. 
# Extract a finishing department record.
df_finishing = df[df['department']=='finishing']
i2 = np.random.randint(df_finishing.shape[0]-1)
selection = df_finishing.iloc[i2:i2+1, :]
selection

# Show that selecting a finishing department record and selecting a sewing department record are two mutually exclusive events.
finishing_and_sewing = np.logical_and(df['department']=='finishing',df['department']=='sewing')
finishing_and_sewing.value_counts()
'''Seen from above there are no records where the department is finishing as well as sewing simultaneously'''

# Probability of selecting finishing and sewing department records simultaneously
P = finishing_and_sewing.sum()/len(df)
print('P(selecting finishing and sewing department records simultaneously)= ', P)
'''Seen that occurrence of both the events simultaneously is zero hence the above mentioned two events are mutually exclusive'''

# A record is selected at random from the dataset. Without replacing it, a second record is selected. 
# Show that getting a finishing department record in the first selection and getting a sewing department record
# in the second selection are dependent events.
'''Hint: Take two cases, one for getting the finishing department and another for not getting the finishing department 
   in the first selection then check if probability for the second selection changes'''

# Case 1:* Getting finishing department record in first selection and sewing department record in the second selection

# count of finishing department records
finishing = df['department'] == 'finishing'
finishing.value_counts()

df_finishing = df[finishing]

# Probability of selecting finishing department record first = count of finishing department records / all records count
P_finishing_first = len(df_finishing) / len(df)
print('P(selecting a finishing department record first)= ', round(P_finishing_first,4))

# Randomly selecting any 'finishing' department record
i = np.random.randint(len(df_finishing)-1)             # -1 is to start the index numbering at 0 instead of 1
selection = df_finishing.iloc[i:i+1, :]                # obtaining a single record with index i
selection