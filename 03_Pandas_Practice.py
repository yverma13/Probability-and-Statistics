import pandas as pd

drinks = pd.read_csv('https://cdn.iisc.talentsprint.com/CDS/Datasets/drinks.csv')
drinks.head()

drinks.groupby('continent').beer_servings.mean().sort_values(ascending=False)

drinks.groupby('continent').wine_servings.describe()

drinks.groupby('continent').wine_servings.mean()

drinks.groupby('continent').wine_servings.median()

drinks.groupby('continent').spirit_servings.agg(['mean','min','max'])
