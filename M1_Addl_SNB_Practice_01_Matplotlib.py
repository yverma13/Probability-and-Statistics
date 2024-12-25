import matplotlib.pyplot as plt

import matplotlib
matplotlib.__version__

import numpy as np
import pandas as pd
X = np.arange(0,100)
Y = X * 2
X = X ** 3

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('X vs Y')
ax.plot(X,Y)
plt.show()

favourite_food_prices = {"Almond butter": 10, "Blueberries": 5, "Eggs": 4, "Cupcakes": 8}
fig, ax = plt.subplots()
ax.bar(favourite_food_prices.keys(), favourite_food_prices.values())

ax.set(title="favourite foods", xlabel="Foods", ylabel="Price ($)")
plt.show()

dictionary1 = {'num':[0,1,2,3,4], 'words':["zero","one","two","three","four"]}
df = pd.DataFrame(dictionary1, index=["zero","one","two","three","four"])
df.iloc[:2,1]

pd.DataFrame(favourite_food_prices, index=[1])

fig, ax = plt.subplots()
ax.barh(list(favourite_food_prices.keys()),list(favourite_food_prices.values()))
ax.set(title="favourite foods",xlabel="Prices($)",ylabel="Food")
plt.show()

X = np.random.randn(1000)
fig, ax = plt.subplots()
ax.hist(X)
plt.show()

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

plt.scatter(x,y)
plt.show()

a = np.array([0, 1, 2, 3, 4, 5])
xx = np.linspace(-0.75,1.,100)


fig, axes = plt.subplots(1,3,figsize=(12,4))

axes[0].scatter(xx, xx + 0.25 * np.random.randn(len(xx)))
axes[0].set_title("Scatter")

axes[1].step(a, a ** 2, lw=2)
axes[1].set_title("Step")

axes[2].bar(a, a**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("Bar")
plt.show()

import pandas as pd
df = pd.read_csv("final.csv", sep="\t")
df.head()

df.country.nunique()

df2k17 = df[df.year == 2007]

# Create a Histogram of life expectancy in the year 2007 across all 142 countries in the given dataset

plt.hist(df2k17['lifeExp'],bins=17)
plt.title('Distribution of Global Life Expectantcy in 2007')
plt.xlabel('Life Expectancy (Years)')
plt.ylabel('# of countries')
plt.show()

# Create a Bar plot showing the per-capita GDP for all the countries in Oceania during 2007

df1 = df2k17[df2k17['continent']=='Oceania']
plt.bar(range(len(df1)),df1['gdpPercap'])
plt.show()

# Plot the Pie chart that displays proportions of all countries contained in each of the 5 continents

countries = df[['country', 'continent']]
country_counts = countries.groupby('continent', as_index=False)['country'].count()
country_counts.columns = ['continent','n_countries']
continents = country_counts['continent']
n_continents = len(country_counts)
plt.pie(country_counts['n_countries'], labels=continents, autopct='%.01f')
plt.title('Proportion of countries per continent.')
plt.show()

# Create a Line plot showing the life expectancy for Spain and Portugal across all the years in the dataset.
portugal = df[df['country'] == 'Portugal']
spain = df[df['country'] == 'Spain']
plt.figure(figsize=(15,5))
plt.plot(spain['year'],spain['lifeExp'],label='Spain')
plt.plot(portugal['year'],portugal['lifeExp'],label='Portugal')
plt.title('Life Expectantcy of Portugal & Spain')
plt.xlabel('Time (Years)')
plt.ylabel('Life Expectancy (Years)')
plt.legend()
plt.show()