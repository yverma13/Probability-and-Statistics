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

# Case 1: Getting finishing department record in first selection and sewing department record in the second selection

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

# As one record is already selected, the total records available becomes one less than total records
df_new = df.drop(selection.index)
'''Essentially I have removed the randomly selected row 'selection' from the original DataFrame, df'''

# count of sewing department records
sewing = df_new['department'] == 'sewing'
sewing.value_counts()

df_sewing = df_new[sewing]

# Probability of selecting sewing department record second = count of sewing department records / (all records count - 1) 
P_sewing_second_given_finishing_first = len(df_sewing)/len(df_new)
print('P(selecting a sewing department record given finishing department record was selected first)= ',round(P_sewing_second_given_finishing_first,4))

P_finishing_sewing = P_finishing_first * P_sewing_second_given_finishing_first
print('P(finishing record first and sewing record second)= ', round(P_finishing_sewing,4))

### Case 2: Getting non-finishing department record in first selection and sewing department record in the second selection

# count of non-finishing department records
non_finishing = df['department']!='finishing'
non_finishing.value_counts()

df_non_finishing = df[non_finishing] 

# Probability of selecting non-finishing department record first = count of non-finishing department records / all records count
P_non_finishing_first = len(df_non_finishing)/len(df)
print('P(selecting a non-finishing department record first)= ', round(P_non_finishing_first,4))

# Randomly selecting any non-'finishing' department record
i = np.random.randint(len(df_non_finishing)-1)
selection = df_non_finishing.iloc[i:i+1, :]
selection

# As one record is already selected, the records available becomes one less than total records
df_new = df.drop(selection.index)

# count of sewing department records
sewing = df_new['department']=='sewing'
sewing.value_counts()

df_sewing = df_new[sewing]

# Probability of selecting sewing department record second = count of sewing department records / (all records count - 1) 
P_sewing_second_given_non_finishing_first = len(df_sewing)/len(df_new)
print('P(selecting a sewing department record given non-finishing department record was selected first)= ', round(P_sewing_second_given_non_finishing_first,4))

P_non_finishing_sewing = P_non_finishing_first * P_sewing_second_given_non_finishing_first
print('P(non-finishing record first and sewing record second)= ', round(P_non_finishing_sewing,4))

# Check for dependency
P_finishing_sewing != P_non_finishing_sewing

# A record is selected among those whose day of week is Monday and also another record is selected among those
# whose day of week is Saturday. Find the probability of getting a finishing department record from the first 
# selection and a sewing department record from the second selection given both events are independent of each other?

# Display different department and day of week
print('Department', df['department'].unique)
print('Day: ', df['day'].unique())

# Select records having day = 'Monday'
df_monday = df[df['day']=='Monday']

P_finishing_from_monday = len(df_monday[df_monday['department']=='finishing']) / len(df_monday)
print('P(selecting finishing department record from Monday records)= ', round(P_finishing_from_monday,4))

# Select records having day = 'Saturday'
df_saturday = df[df['day']=='Saturday']

P_sewing_from_saturday = len(df_saturday[df_saturday['department']=='sewing'])/len(df_saturday)
print('P(selecting sewing department record from Saturday records)= ', round(P_sewing_from_saturday,4))

# As events are independent,
P_finishing_and_sewing = P_finishing_from_monday * P_sewing_from_saturday
print('P(getting finishing department from first selection and sewing department from second selection)= ', round(P_finishing_and_sewing,4))


# Let  S  is the sample space given below and corresponding  P(X=xi)  is also given, where  X  is a 
# discrete random variable. Find the probability at  X=0.

df1 = pd.DataFrame({'X=0': '?', 'X=1':0.2, 'X=3':0.3, 'X=4': 0.1}, index= ['P(X=xi)'])
df1

# For a discrete random variable we know that sum of all P(X=xi) = 1,
df1['X=0'] = 1 - sum(df1.iloc[0,1:])
df1

# Plot the PMF of the discrete random variable X defined as total number of heads while tossing a coin thrice.

# Our sample space would consist of {HHH, HHT,HTH, THH, TTH, THT, HTT, TTT}
X = [0,1,2,3] #number of heads we can get

P_X0 = 1/8   # P(X=0)     {TTT}
P_X1 = 3/8   # P(X=1)     {HTT, THT, TTH}
P_X2 = 3/8   # P(X=2)     {HHT, HTH, THH}
P_X3 = 1/8   # P(X=3)     {HHH}

P_Xi = [P_X0, P_X1, P_X2, P_X3]

# Plotting PMF
sns.barplot(x=X, y=P_Xi)
plt.title('PMF'); plt.xlabel('Number of Heads'); plt.ylabel('Probability')
plt.show()

# Plotting CDF or cumulative distribution function
sns.barplot(x=X, y=np.array(P_Xi).cumsum())
plt.title('Cumulative Distribution Function'); plt.xlabel('Number of heads'); plt.ylabel('Cumulative Probability')
plt.show()

# Compute the value of  P(1<X<2) . Such that the density function is given by, f(x)={kx^3 for 0≤x≤3, 0 otherwise 
# Also, plot the PDF and CDF for random variable  X

# ∫ f(x) dx = 1
# Using the above property we find k,
# ∫ (k*x**3)dx = 1
# k = 1 / ∫ (x**3)dx

k = 1 / (integrate.quad(lambda x: x**3,0,3)[0])
print('k= ', round(k,4))

# Now the probability density for 1<X<2 is given by,
P = integrate.quad(lambda x: k*x**3, 1, 2)[0]
print('P(1<X<2)= ', round(P,4))

# Create 100 values within 0 to 3 in order to plot PDF and CDF
x = np.linspace(0,3,100)
df2 = pd.DataFrame({'X':[], 'PDF':[], 'CDF':[]})
df2['X'] = x
df2['PDF'] = df2['X'].apply(lambda v: k*v**3)
df2['CDF'] = df2['X'].apply(lambda v: integrate.quad(lambda u: k*u**3, 0, v)[0])
df2.head()

# Plotting PDF
sns.lineplot(x='X', y='PDF', data=df2)
plt.title('PDF'); plt.xlabel('X'); plt.ylabel('Probability density')
plt.show()

# Plotting CDF
sns.lineplot(x='X', y='CDF', data=df2)
plt.title('Cummulative distribution function'); plt.xlabel('X'); plt.ylabel('Probability density')
plt.show()

## Approach without using 'integrate' ##

# Defining the density function (given as per the question)
def f(x):
    if 0 <= x <= 3:
        return x**3
    else:
        return 0 
    
# Numerically computing the integral to find k
def compute_k():
    k_approx = 0 
    dx = 0.001  # Small interval for numerical integration
    x_values = np.arange(0,3,dx)
    for x in x_values:
        k_approx += f(x)*dx 
    return 1/k_approx

# Computing k
k = compute_k()
print("Value of k: ", k)

# Computing the probability P(1 < X < 2)
P_1_to_2 = 0
dx = 0.001
x_values = np.arange(1,2,dx)
for x in x_values:
    P_1_to_2 += k*f(x)*dx

print("Probability P(1 < X < 2):", P_1_to_2)

# Plotting the PDF
import matplotlib.pyplot as plt
import numpy as np

x_values = np.linspace(0,3,1000)
y_values = [f(x) for x in x_values]
plt.plot(x_values, y_values)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Probability Density Function (PDF)")
plt.grid(True)
plt.show()

# Computing the CDF
cdf_values = []
cumulative_prob = 0
for x in x_values:
    cumulative_prob += f(x)*dx
    cdf_values.append(cumulative_prob)
print("CDF:", cdf_values)

# Plotting the CDF
plt.plot(x_values, cdf_values)
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function (CDF)")
plt.grid(True)
plt.show()