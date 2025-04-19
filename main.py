import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skn
import plotly.express as px
import seaborn as sns

from scipy.stats import ttest_ind
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings('ignore')

# Streamlit
import streamlit as st

df = pd.read_csv('./NY-House-Dataset.csv')
set_cols = ['TYPE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'PRICE']
df = df[set_cols]

st.set_page_config(layout="wide")
st.title("CAP2757 - Final Project - NY Housing Data")


# Expander of raw data toggle
with st.expander("View Raw Data"):
    st.dataframe(df)
# Display dataset preview in streamlit
st.write(df.head())

df.TYPE.unique()
df.TYPE.value_counts()
keep_types = ['Co-op for sale', 'House for sale', 'Condo for sale', 'Multi-family home for sale']
df = df[df.TYPE.isin(keep_types)]
df.describe()

lower_iqr = np.nanpercentile(df.PRICE, 10)
upper_iqr = np.nanpercentile(df.PRICE, 90)

iqr = upper_iqr - lower_iqr

lower_bound = lower_iqr - (1.5 * iqr)
upper_bound = upper_iqr + (1.5 * iqr)

df = df[(df['PRICE'] >= lower_bound) & (df['PRICE'] <= upper_iqr)]

house_types = df.TYPE.unique()
fig, axs = plt.subplots(2,2, figsize = (10, 10))

axs = axs.flatten()

for i in range(len(house_types)):
    sns.histplot(
            data=df[df['TYPE'] == house_types[i]],
            x='PRICE',
            kde=True,
            ax=axs[i]
            )
    axs[i].set_title(f"House type: {house_types[i]}")

plt.tight_layout()
plt.show()

# T-Test 
house_types = df['TYPE'].unique()
res = []

# Compare all pairs of house types
for a, b in itertools.combinations(house_types, 2):
    prices_a = df[df['TYPE'] == a]['PRICE']
    prices_b = df[df['TYPE'] == b]['PRICE']
    t_stat, p_val = ttest_ind(prices_a, prices_b, equal_var=False) # Welch's t-test
    temp = {
            "comparison": f"{a} vs {b}",
            "t-stat": t_stat,
            "p-value": p_val
            }
    res.append(temp)

res = pd.DataFrame(res)

res

df.shape[0]









