import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import random

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000
    banks = ['HDFC', 'ICICI', 'SBI', 'Axis', 'Kotak']
    regions = ['North', 'South', 'East', 'West', 'Central']
    transaction_types = ['NEFT', 'RTGS', 'IMPS', 'UPI', 'SWIFT']
    statuses = ['Completed', 'Completed', 'Completed', 'Pending', 'Failed']
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(n)]
    amounts = np.random.exponential(scale=50000, size=n)
    amounts[np.random.choice(n, 50)] *= 20
    return pd.DataFrame({
        'transaction_id': [f'TXN{str(i).zfill(5)}' for i in range(n)],
        'date': dates,
        'bank': np.random.choice(banks, n),
        'region': np.random.choice(regions, n),
        'transaction_type': np.random.choice(transaction_types, n),
        'amount': np.round(amounts, 2),
        'status': np.random.choice(statuses, n),
        'processing_time_hrs': np.round(np.random.exponential(2, n), 2)
    })

st.set_page_config(page_title="Regional Report", page_icon="🗺️", layout="wide")
st.title("🗺️ Regional Performance Report")

df = load_data()

regional = df.groupby('region').agg(
    total_transactions=('transaction_id','count'),
    total_volume=('amount','sum'),
    avg_amount=('amount','mean'),
    completion_rate=('status', lambda x: (x=='Completed').mean()*100),
    avg_processing=('processing_time_hrs','mean')
).reset_index()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
regional['performance_tier'] = kmeans.fit_predict(regional[['total_volume','completion_rate']])
tier_map = {int(regional.loc[regional['total_volume'].idxmax(), 'performance_tier']): 'High Performer',
            int(regional.loc[regional['total_volume'].idxmin(), 'performance_tier']): 'Needs Attention'}
regional['performance_tier'] = regional['performance_tier'].apply(lambda x: tier_map.get(x, 'Developing'))

st.subheader("Regional Performance Summary")
st.dataframe(regional, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(regional, x='region', y='total_volume', color='performance_tier', title='Total Volume by Region')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(regional, x='region', y='completion_rate', color='performance_tier', title='Completion Rate by Region (%)')
    st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(regional, x='total_volume', y='completion_rate', color='performance_tier',
                 text='region', size='total_transactions', title='Performance Tier Clustering')
st.plotly_chart(fig, use_container_width=True)
