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

# Check datatype of each column
playstore_data.dtypes

### FIND OUT THE NON-ENGLISH APPS ###

def is_English(string):
    spl_count = 0
    for character in string:
        if ord(character) > 127:
            spl_count += 1
    if spl_count > len(string) // 2:
        return False
    return True

# Find the Non-English Apps
playstore_data[~playstore_data['App'].apply(is_English)]

# Filter the Non English Apps
playstore_data = playstore_data[playstore_data['App'].apply(is_English)]
playstore_data.shape

# In the size column, multiply 1000,000 with M in the cell and multiply by 1000 if we have K in the cell

playstore_data.Size.value_counts()

playstore_data ['Size'] = playstore_data ['Size'].apply(lambda x: str(x).replace('Varies with device','NaN') if 'Varies with device' in x else x)
playstore_data ['Size'] = playstore_data ['Size'].apply(lambda x: float(str(x).rstrip('M'))*(10**6) if 'M' in str(x) else x)
playstore_data['Size'] = playstore_data['Size'].apply(lambda x: float(str(x).rstrip('k'))*(10**3) if 'k' in str(x) else x)
playstore_data = playstore_data[~(playstore_data['Size'] == 'NaN')]
playstore_data['Size'] = playstore_data['Size'].astype(float)


''' Task 2: Visualisation '''

# Find the number of apps in various categories by using an appropriate plot

playstore_data['Category'].nunique()

count_of_Apps = playstore_data['Category'].value_counts()
count_of_Apps

count_of_Apps.index.values

plt.figure(figsize=(10,15))
plt.pie(count_of_Apps, labels = count_of_Apps.index.values, autopct='%1.1f%%')
plt.show()

# Explore the distribution of free and paid apps across different categories

free_apps = playstore_data[playstore_data.Type == 'Free']
paid_apps = playstore_data[playstore_data.Type == 'Paid']
paid_apps.shape, free_apps.shape

paid_categories = paid_apps['Category'].value_counts()
free_categories = free_apps['Category'].value_counts()
paid_categories

len(free_categories), len(paid_categories)

N = 10

idx = np.arange(N)

p1 = plt.bar(idx, free_categories.values[:10])
p2 = plt.bar(idx, paid_categories.values[:10], bottom = free_categories.values[:10])

plt.xticks(idx, free_categories.index[:10], rotation=35)
plt.legend((p1[0], p2[0]),('Free', 'Paid'))
plt.show()

# Represent the distribution of app rating on a scale of 1-5 using an appropriate plot

ratings = playstore_data['Rating']

plt.hist(ratings, bins=5)
plt.title('Rating Distribution')
plt.xlabel('Ratings')
plt.show()

# Identify outliers of the rating column by plotting the boxplot category wise and handle them.

df_categories = playstore_data.groupby('Category').filter(lambda x: len(x) >= 120)

sns.boxplot(y=df_categories.Rating, x=df_categories.Category, data=playstore_data);
plt.xticks(rotation=50)
plt.xlabel('Categories',fontsize=17,fontweight='bold',color='#191970', )
plt.ylabel('Ratings', fontsize=17, fontweight='bold',color='#191970')
plt.show()

def remove_outliers(data):
    data_mean, data_std = data.mean(), data.std()
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers_removed = [x if x > lower and x < upper else data_mean for x in data]
    return outliers_removed

playstore_data['Rating'] = remove_outliers(playstore_data['Rating'])

df_categories = playstore_data.groupby('Category').filter(lambda x: len(x) >= 120)

sns.boxplot(y=df_categories.Rating, x=df_categories.Category,data=playstore_data);
plt.xticks(rotation=50)
plt.xlabel('Categories',fontsize=17, fontweight='bold', color='#191970', )
plt.ylabel('Ratings', fontsize=17, fontweight='bold', color='#191970')

# Plot the barplot of all the categories indicating no. of installs

playstore_data['Installs'] = playstore_data['Installs'].str.rstrip('+').str.replace(',','')
playstore_data['Installs'] = playstore_data['Installs'].astype(int)

temp_df = playstore_data.groupby(['Category']).agg({'Installs':'sum'}).sort_values(by='Installs',ascending=False).reset_index()

sns.barplot(x=temp_df['Installs'],y=temp_df['Category'])
plt.yticks(rotation=10)
plt.xlabel('Installs',fontsize=15,color='#191970')
plt.ylabel('Categories', fontsize=15, color='#191970')
plt.show()


''' Task 3: Insights '''

# Does price correlate with the size of the app? We will see that they do not correlate!

playstore_data['Price'].unique()

playstore_data['Price'] = playstore_data['Price'].str.lstrip('$')
playstore_data['Price'] = playstore_data['Price'].astype(float)

sns.lmplot(x='Price', y='Size', data=playstore_data, fit_reg=False)
plt.show()

# Find the popular app categories based on rating and no. of installs

popular_categories = playstore_data.groupby(['Category']).agg({'Installs':'sum','Rating':'sum'}).sort_values(by='Rating',ascending=False).reset_index()
popular_categories.head()

# Average rating

popular_categories1 = playstore_data.groupby(['Category']).Rating.mean().sort_values(ascending=False).reset_index()
popular_categories1

# How many apps are produced in each year category-wise?

playstore_data["Year"] = playstore_data['Last Updated'].str[-4:]
playstore_data["Year"].unique()

App2018 = playstore_data["Year"]=="2018"

plt.title('Downloads in 2018')
plt.xticks(rotation = 'vertical')
sns.countplot(hue = 'Year', x = 'Category', data = App2018)

App2017 = playstore_data[playstore_data["Year"]== "2017"]
plt.title('Downloads in 2017')
plt.xticks(rotation = 'vertical')
sns.countplot(hue = 'Year', x = 'Category', data = App2017)

# Identify the highest paid apps with a good rating

topRated = playstore_data[(playstore_data.Rating > 4.0) & (playstore_data.Type == 'Paid')].sort_values(by='Price',ascending=False)
topRated['Reviews'].head()

# Are the top-rated apps genuine ? How about checking reviews count of top-rated apps ?

topRated = playstore_data[playstore_data.Rating == playstore_data.Rating.max()]
idx_topRate = np.arange(0, len(topRated))

topRated['Reviews'] = topRated['Reviews'].astype(int)
topRated['Reviews']

topRated['Reviews'].max(), topRated['Reviews'].min()

plt.title("Distribution of Review count for top-rated apps")
plt.plot(idx_topRate, topRated['Reviews'])
plt.show()

# Frequency distribution of Reviews count
sns.displot(playstore_data[playstore_data.Rating == playstore_data.Rating.max()].Reviews)
plt.show()

# If the number of reviews of an app is very low, what could be the reason for its top-rating?

Apps_Below_review_5 = topRated[topRated['Reviews'] < 5]
Free_apps_below_ReviewCount5 = Apps_Below_review_5[Apps_Below_review_5['Type'] == 'Free'].shape[0]
Paid_apps_below_ReviewCount5 = Apps_Below_review_5[Apps_Below_review_5['Type'] == 'Paid'].shape[0]
Free_apps_below_ReviewCount5 , Paid_apps_below_ReviewCount5

# Conclusion: Most of the top-rated and less reviews are free, that why user rated 5.0