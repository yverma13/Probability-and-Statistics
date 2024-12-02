# Step 1: Import the necessary libraries
import pandas as pd 

# Step 2: Assign it to a variable called users and use the 'user_id' as index
users = pd.read_csv('https://cdn.iisc.talentsprint.com/CDS/Datasets/u.user', sep='|', index_col='user_id')

# Step 3: See the first 25 entries
users.head(25)

# Step 4: See the last 10 entries
users.tail(10)

# Step 5: What is the number of observations in the dataset?
users.shape[0]

# Step 6: What is the number of columns in the dataset?
users.shape[1]

# Step 7: Print the name of all the columns
users.columns

# Step 8: How is the dataset indexed?
# "the index" (aka "the labels")
users.index

# Step 9: What is the data type of each column?
users.dtypes

# Step 10: Print only the occupation column
users.occupation
#or
users['occupation']

# Step 11: How many different occupations are in this dataset?
users.occupation.nunique()
#or by using value_counts() which returns the count of unique elements
#users.occupation.value_counts().count()

# Step 12: What is the most frequent occupation?
#Because "most" is asked
users.occupation.value_counts().head(1).index[0]
#or to have the top 5
# users.occupation.value_counts().head()

# Step 13: Summarize the DataFrame
users.describe() #Notice: by default, only the numeric columns are returned.

# Step 14: Summarize all the columns
users.describe(include = "all") #Notice: By default, only the numeric columns are returned.

# Step 15: Summarize only the occupation column
users.occupation.describe()

# Step 16: What is the mean age of users?
round(users.age.mean())

# Step 17: What is the age with least occurrence?
users.age.value_counts().tail()