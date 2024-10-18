import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Songs.csv")

print(df.shape) # check to see current rows and cols in the data set 


## Cleaning the data 
##print (pd.isnull(df).sum())
df.dropna(inplace= True) # drop the null values 





print(df.columns)

ax = sns.countplot (x = 'released_month', data = df)

for bars in ax.containers:
    ax.bar_label(bars)

plt.show()