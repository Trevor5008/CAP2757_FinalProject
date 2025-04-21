import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skn
import plotly.express as px
import seaborn as sns
import pydeck as pdk

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
from stqdm import stqdm

df = pd.read_csv('NY-House-Dataset.csv')
set_cols = ['TYPE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE', 'PRICE']
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

# Histogram figure
fig_hist = make_subplots(rows=3, cols=2, subplot_titles=house_types)
row_idx, col_idx = 1,1

for i in range(len(house_types)):
    if row_idx > 3:
        row_idx = 1
        col_idx += 1
    fig_hist.add_trace(go.Histogram(x=df[df.TYPE == house_types[i]]['PRICE'], name=house_types[i]), row=row_idx, col=col_idx)
    row_idx += 1
    
fig_hist.update_layout(title_text='Price Distribution by House Type', height=800, width=1000)

# Render Histograms
st.subheader("Histogram of Price by House Type", divider="grey")
st.plotly_chart(fig_hist, use_container_width=True)

# Scatter plot
fig = px.scatter(data_frame=df, x='PROPERTYSQFT', y='PRICE', facet_col='TYPE', trendline='ols')
# Render Scatter plots
st.subheader("Property Size vs. Price Scatterplot", divider="grey")
st.plotly_chart(fig, use_container_width=True)

# Box plot for each house type
fig = px.box(data_frame=df, x='TYPE', y='PRICE', color='TYPE')

# Display box plot
st.subheader("Box plots per House Type", divider="grey")
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
st.subheader("Box plots per Bath group", divider="grey")
st.plotly_chart(fig, use_container_width=True)

bath_groups = sorted(df['BATH_GROUP'].unique())

# Bath groups unique data display
st.subheader(":bath: Bath Groups", divider="grey")
for group in bath_groups: 
    with st.expander(f"Group {group}"):
        unique_baths = df[df['BATH_GROUP'] == group]['BATH'].unique().astype(str)
        st.write(f"Includes bath counts: {', '.join(unique_baths)}")

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
st.subheader("Box plots per Beds group", divider="grey")
st.plotly_chart(fig, use_container_width=True)

bed_groups = sorted(df['BEDS_GROUP'].unique())

# Bed groups unique data display
st.subheader(":bed: Bed Groups", divider="grey")
for group in bed_groups: 
    with st.expander(f"Group {group}"):
        unique_beds = df[df['BEDS_GROUP'] == group]['BEDS'].unique().astype(str)
        st.write(f"Includes bed counts: {', '.join(unique_beds)}")

df['combined_cols'] = df['BEDS_GROUP'].astype(str) + df['BATH_GROUP'].astype(str) + df['TYPE'].astype(str)
value_counts = df['combined_cols'].value_counts()
df = df[df['combined_cols'].isin(value_counts[value_counts >= 8].index)]

categorical_cols = ['BEDS_GROUP', 'BATH_GROUP','TYPE']

X = df.drop(columns = ['PRICE'])
y = np.log(df['PRICE'])

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X[col] = X[col].cat.codes


# 3D plot of lat/long locations
st.subheader(":world_map: Map of Property Listings (NY Area)", divider="grey")
# drop n/a vals
map_df = df[['LATITUDE', 'LONGITUDE', 'TYPE', 'PRICE']].dropna().copy()
price_min = map_df['PRICE'].min()
price_max = map_df['PRICE'].max()

map_df['price_scaled'] = (map_df['PRICE'] - price_min) / (price_max - price_min)

# Color gradient defined
map_df['R'] = (map_df['price_scaled'] * 256).astype(int)
map_df['G'] = 30 
map_df['B'] = (255 - map_df['R']).astype(int) 
map_df['A'] = 160

map_df['COLOR'] = map_df[['R', 'G', 'B', 'A']].values.tolist()

layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[LONGITUDE, LATITUDE]',
        get_color='COLOR',
        get_radius=100,
        pickable=True
        )

view_state = pdk.ViewState(
        latitude=40.7128, longitude=-74.0060,
        zoom=9,
        pitch=30
        )

# Render 3d chart
col1, col2 = st.columns([6,1]) 

with col1:
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{TYPE}\n${PRICE}"}))
with col2:
    st.markdown("#### Color Legend")
    st.markdown("""
    <div style="width: 100%; padding-top: 10px;">
        <div style="height: 20px; width: 100%; background: linear-gradient(to right, blue, red); border-radius: 5px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 14px;">
            <span style="color: #777;">Low</span>
            <span style="color: #777;">High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Train/Test slider control
st.subheader("Test Size Allocation (%)")
test_size = st.select_slider(label="", value=25, options=[i for i in range(5,35,5)])

def make_scaler(feature):
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature.values.reshape(-1, 1))
    return feature , scaler

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

models = [
         LinearRegression(),
         DecisionTreeRegressor(criterion= 'squared_error' ,  random_state=rs),
         RandomForestRegressor( n_estimators=100,criterion= 'friedman_mse',random_state=rs),
         KNeighborsRegressor(n_neighbors=5 , weights='distance' , algorithm='auto')
         ]

# Spinner 
with st.spinner("Preparing data..."):

    X['Area'] = X['PROPERTYSQFT']
    X['Area_sqrt'] = np.sqrt(X['Area'])
    X['Area_log'] = np.log(X['Area'])
    X['Area_2'] = X['Area'] ** 2
    X['Area_3'] = X['Area'] ** 3


    X['Lat'] = X['LATITUDE']
    X['Lat_2'] = X['Lat'] ** 2
    X['Lat_3'] = X['Lat'] ** 3

    X['Lon'] = X['LONGITUDE']
    X['Lon_2'] = X['Lon'] ** 2
    X['Lon_3'] = X['Lon'] ** 3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify=X['combined_cols'])

    X_train['Area'] , area_scaler = make_scaler(X_train['Area'])
    X_train['Area_sqrt'] , area_sqrt_scaler = make_scaler(X_train['Area_sqrt'])
    X_train['Area_log'] , area_log_scaler = make_scaler(X_train['Area_log'])
    X_train['Area_2'] , area_2_scaler = make_scaler(X_train['Area_2'])
    X_train['Area_3'] , area_3_scaler = make_scaler(X_train['Area_3'])

    X_test['Area'] = area_scaler.transform(X_test['Area'].values.reshape(-1, 1))
    X_test['Area_sqrt'] = area_sqrt_scaler.transform(X_test['Area_sqrt'].values.reshape(-1, 1))
    X_test['Area_log'] = area_log_scaler.transform(X_test['Area_log'].values.reshape(-1, 1))
    X_test['Area_2'] = area_2_scaler.transform(X_test['Area_2'].values.reshape(-1, 1)) 
    X_test['Area_3'] = area_3_scaler.transform(X_test['Area_3'].values.reshape(-1, 1))

    X_train['Lat'] , lat_scaler = make_scaler(X_train['Lat'])
    X_train['Lat_2'] , lat_2_scaler = make_scaler(X_train['Lat_2'])
    X_train['Lat_3'] , lat_3_scaler = make_scaler(X_train['Lat_3'])

    X_test['Lat'] = lat_scaler.transform(X_test['Lat'].values.reshape(-1, 1))
    X_test['Lat_2'] = lat_2_scaler.transform(X_test['Lat_2'].values.reshape(-1, 1)) 
    X_test['Lat_3'] = lat_3_scaler.transform(X_test['Lat_3'].values.reshape(-1, 1))

    X_train['Lon'] , lon_scaler = make_scaler(X_train['Lon'])
    X_train['Lon_2'] , lon_2_scaler = make_scaler(X_train['Lon_2'])
    X_train['Lon_3'] , lon_3_scaler = make_scaler(X_train['Lon_3'])

    X_test['Lon'] = lon_scaler.transform(X_test['Lon'].values.reshape(-1, 1))
    X_test['Lon_2'] = lon_2_scaler.transform(X_test['Lon_2'].values.reshape(-1, 1)) 
    X_test['Lon_3'] = lon_3_scaler.transform(X_test['Lon_3'].values.reshape(-1, 1))

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)) 
    y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

    drop_cols = ['BEDS' , 'BATH' , 'PROPERTYSQFT' , 'combined_cols' , 'LATITUDE' , 'LONGITUDE']

    X_train = X_train.drop(columns = drop_cols)
    X_test = X_test.drop(columns = drop_cols)

    rf = RandomForestRegressor(
        n_estimators=100,
        criterion= 'friedman_mse',
        random_state=rs
    )

    rf.fit(X_train, y_train)
    st.subheader("Random Forest Feature Importance")
    feat_imp = pd.DataFrame([X_train.columns.transpose() , rf.feature_importances_.transpose()] , index = ['feature' , 'importance']).transpose().sort_values(by = 'importance' , ascending = False)
    feat_imp['cumulative_importance'] = feat_imp['importance'].cumsum()
    st.dataframe(feat_imp)

    st.subheader("Selected Features:")   
    model_features = feat_imp[feat_imp['cumulative_importance'] <= 0.65]['feature'].values.tolist()
    st.dataframe(model_features)
    results = [train_and_evaluate_model(model, y_scaler, X_train[model_features], y_train, X_test[model_features], y_test) for model in models]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{result['model_name']}" for result in results])

    row_idx = 1
    col_idx = 1

    y_train = np.exp(y_scaler.inverse_transform(y_train.reshape(-1, 1)))


    for _ , result in enumerate(results):
    
        if row_idx > 2:
            row_idx = 1
            col_idx += 1
    
        y_train_pred = np.exp(result['y_train_pred'])
        res = y_train_pred - y_train

        fig.add_trace(
            go.Scatter(
                x=y_train_pred.ravel(),
                y=res.ravel(),
                mode='markers' ,
                name = result['model_name'] ,
                hovertext=y_train.ravel() ,
                hoverinfo='text'),
            row=row_idx, col=col_idx
            )
        fig.update_yaxes(title_text="Residual", row=row_idx, col=col_idx)
        fig.update_xaxes(title_text="Predicted Price", row=row_idx, col=col_idx)
        row_idx += 1

    fig.update_layout(
        height=1200,  
        width=1300,
        xaxis_title="Predicted Price",
        yaxis_title="Residuals",  
        title_text="Residual Plots for each model"
    )


st.plotly_chart(fig, use_container_width=True)

st.subheader(":chart_with_upwards_trend: Model Performance Metrics", divider="grey")
perf_df = pd.DataFrame([{
    'Model': r['model_name'],
    'RMSE': r['test_rmse'],
    'MAPE': r['test_mape']
    } for r in results])
st.dataframe(perf_df)
