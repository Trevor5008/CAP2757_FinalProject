import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skn
import plotly.express as px
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings('ignore')

rs = 42

# Streamlit
import streamlit as st

df = pd.read_csv('NY-House-Dataset.csv')
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

# Cleaning
df['TYPE'] = df['TYPE'].str.replace(' for sale', '')
df.loc[df['TYPE'] == 'Condop', 'TYPE'] = 'Condo'

keep_types = ['Co-op', 'House', 'Condo', 'Multi-family home']
df.loc[~(df.TYPE.isin(keep_types)), 'TYPE'] = 'Other'

# Describe dataset
st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

st.subheader("Value Counts for TYPE")
st.dataframe(df.TYPE.value_counts())


lower_iqr = np.nanpercentile(df.PRICE, 10)
upper_iqr = np.nanpercentile(df.PRICE, 90)

iqr = upper_iqr - lower_iqr

lower_bound = lower_iqr - (1.5 * iqr)
upper_bound = upper_iqr + (1.5 * iqr)

df = df[(df['PRICE'] >= lower_bound) & (df['PRICE'] <= upper_iqr)]

house_types = df.TYPE.unique()

fig = make_subplots(rows=3, cols=2, subplot_titles=house_types)
row_idx, col_idx = 1,1

for i in range(len(house_types)):
    if row_idx > 3:
        row_idx = 1
        col_idx += 1
    fig.add_trace(go.Histogram(x=df[df.TYPE == house_types[i]]['PRICE'], name=house_types[i]), row=row_idx, col=col_idx)
    row_idx += 1

fig.update_layout(title_text='Price Distribution by House Type', height=800, width=1000)
fig = px.scatter(data_frame=df, x='PROPERTYSQFT', y='PRICE', facet_col='TYPE', trendline='ols')

# Render in streamlit
st.plotly_chart(fig, use_container_width=True)

# Box plot
fig = px.box(data_frame=df, x='TYPE', y='PRICE', color='TYPE')
# Display box plot
st.plotly_chart(fig, use_container_width=True)

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

rejected, pvals_corrected, _, _ = multipletests(res['p-value'], alpha=0.05, method='bonferroni')
res['p_value_corrected'] = pvals_corrected
res['rejected'] = rejected

res

# Bathroom category comparison
bath_price = df.groupby('BATH')['PRICE'].mean().reset_index()

# Reshape price for clustering
X = bath_price['PRICE'].values.reshape(-1,1)

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=rs).fit(X)
bath_price['BATH_GROUP'] = kmeans.labels_

# Map back
bath_group_map = dict(zip(bath_price['BATH'], bath_price['BATH_GROUP']))
df['BATH_GROUP'] = df['BATH'].map(bath_group_map)

fig = px.box(data_frame=df, x='BATH_GROUP', y='PRICE', color='BATH_GROUP')

# Render box plot
st.plotly_chart(fig, use_container_width=True)

bath_groups = sorted(df['BATH_GROUP'].unique())

for group in bath_groups: 
    st.write(f"Group {group}:")
    st.write(f"Unique: {df[df['BATH_GROUP'] == group]['BATH'].unique().astype(str)}")

# Get average price per bed type
bed_price = df.groupby('BEDS')['PRICE'].mean().reset_index()

# Reshape price for clustering
X = bed_price['PRICE'].values.reshape(-1,1)

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=rs).fit(X)
bed_price['BEDS_GROUP'] = kmeans.labels_

# Map back
bed_group_map = dict(zip(bed_price['BEDS'], bed_price['BEDS_GROUP']))
df['BEDS_GROUP'] = df['BEDS'].map(bed_group_map)

fig = px.box(data_frame=df, x='BEDS_GROUP', y='PRICE', color='BEDS_GROUP')

# Render box plot for beds group
st.plotly_chart(fig, use_container_width=True)

bed_groups = sorted(df['BEDS_GROUP'].unique())

for group in bath_groups:
    st.write(f"Group {group}")
    st.write(f"Unique bed values: {df[df['BEDS_GROUP'] == group]['BEDS'].unique().astype(str)}")

categorical_cols = ['BEDS_GROUP', 'BATH_GROUP','TYPE']

X = df.drop(columns = ['PRICE'])
y = np.log(df['PRICE'])

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X[col] = X[col].cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

x_scaler = MinMaxScaler()
X_train['PROPERTYSQFT'] = x_scaler.fit_transform(X_train['PROPERTYSQFT'].values.reshape(-1,1))
X_test['PROPERTYSQFT'] = x_scaler.transform(X_test['PROPERTYSQFT'].values.reshape(-1,1))

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = y_scaler.transform(y_test.values.reshape(-1,1))

st.write(f"X_train shape: {X_train.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_test shape: {y_test.shape}")

# Model training and evaluation
def train_and_evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1,1))

    y_test_pred = model.predict(X_test)
    y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    ret = {
            'model_name': type(model).__name__,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'test_rmse': rmse,
            'test_mape': mape
            }

    return ret

# Ensemble 
models = [
        LinearRegression(),
        DecisionTreeRegressor(criterion='squared_error', max_depth=5),
        RandomForestRegressor(criterion='squared_error', max_depth=5),
        KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')
        ]

results = [train_and_evaluate_model(model, y_scaler, X_train, y_train, X_test, y_test) for model in models]

fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=[f"{result['model_name']}" for result in results])
row_idx = 1
col_idx = 1

y_train = np.exp(y_scaler.inverse_transform(y_train.reshape(-1,1)))

for _, result in enumerate(results):
    if row_idx > 2:
        row_idx = 1
        col_idx += 1

    y_train_pred = np.exp(result['y_train_pred'])
    res = y_train_pred - y_train

    fig.add_trace(go.Scatter(x=y_train_pred.ravel(), y=res.ravel(), mode='markers', name=result['model_name'], hovertext=y_train.ravel(), hoverinfo='text'), row=row_idx, col=col_idx)
    row_idx += 1

fig.update_layout(
        height=1200,
        width=1300,
        xaxis_title="Predicted Price",
        yaxis_title="Residuals",
        title_text="Residual Plots for each model"
        )

st.plotly_chart(fig, use_container_width=True)

# Y test 
y_test = np.exp(y_scaler.inverse_transform(y_test.reshape(-1,1)))

fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{result['model_name']} RMSE = {result['test_rmse']:.2f}" for result in results])

row_idx = 1
col_idx = 1


for _, result in enumerate(results):
    if row_idx > 2:
        row_idx = 1
        col_idx += 1

    y_test_pred = np.exp(result['y_test_pred'])

    fig.add_trace(go.Scatter(x=y_test.ravel(), y=y_test_pred.ravel(), mode='markers', name=result['model_name']), row=row_idx, col=col_idx)

    row_idx += 1

fig.update_layout(
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        height=1200,
        width=1300,
        title_text='Predicted vs actual prices on test set'
        )

st.plotly_chart(fig, use_container_width=True)
