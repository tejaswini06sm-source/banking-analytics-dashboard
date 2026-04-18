import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        'date': dates, 'bank': np.random.choice(banks, n),
        'region': np.random.choice(regions, n),
        'transaction_type': np.random.choice(transaction_types, n),
        'amount': np.round(amounts, 2),
        'status': np.random.choice(statuses, n),
        'processing_time_hrs': np.round(np.random.exponential(2, n), 2)
    })

st.set_page_config(page_title="Regional Report", page_icon="🗺️", layout="wide")
st.title("🗺️ Regional Performance Report")
st.markdown("K-Means clustering to segment regions into performance tiers and identify operational gaps.")
st.markdown("---")

df = load_data()

st.sidebar.title("🔎 Filters")
selected_type = st.sidebar.multiselect("Transaction Type", df['transaction_type'].unique(), default=list(df['transaction_type'].unique()))
df = df[df['transaction_type'].isin(selected_type)]

regional = df.groupby('region').agg(
    total_transactions=('transaction_id','count'),
    total_volume=('amount','sum'),
    avg_amount=('amount','mean'),
    completion_rate=('status', lambda x: (x=='Completed').mean()*100),
    failed_rate=('status', lambda x: (x=='Failed').mean()*100),
    avg_processing=('processing_time_hrs','mean')
).reset_index()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
regional['cluster'] = kmeans.fit_predict(regional[['total_volume','completion_rate','avg_processing']])
tier_map = {int(regional.loc[regional['total_volume'].idxmax(), 'cluster']): 'High Performer',
            int(regional.loc[regional['total_volume'].idxmin(), 'cluster']): 'Needs Attention'}
regional['Performance Tier'] = regional['cluster'].apply(lambda x: tier_map.get(x, 'Developing'))

col1, col2, col3 = st.columns(3)
hp = regional[regional['Performance Tier']=='High Performer']['region'].values
na = regional[regional['Performance Tier']=='Needs Attention']['region'].values
dev = regional[regional['Performance Tier']=='Developing']['region'].values
col1.success(f"🏆 High Performer: {', '.join(hp)}")
col2.warning(f"🔧 Developing: {', '.join(dev)}")
col3.error(f"⚠️ Needs Attention: {', '.join(na)}")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(regional.sort_values('total_volume', ascending=False),
                 x='region', y='total_volume', color='Performance Tier',
                 title='💰 Total Volume by Region', text_auto='.2s',
                 color_discrete_map={'High Performer':'#00CC96','Developing':'#FFA15A','Needs Attention':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(regional, x='region', y='completion_rate', color='Performance Tier',
                 title='✅ Completion Rate by Region (%)', text_auto='.1f',
                 color_discrete_map={'High Performer':'#00CC96','Developing':'#FFA15A','Needs Attention':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(regional, x='total_volume', y='completion_rate',
                     color='Performance Tier', text='region', size='total_transactions',
                     title='🎯 Performance Tier Clustering',
                     color_discrete_map={'High Performer':'#00CC96','Developing':'#FFA15A','Needs Attention':'#EF553B'})
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(regional, x='region', y='avg_processing', color='Performance Tier',
                 title='⏱️ Avg Processing Time by Region',
                 color_discrete_map={'High Performer':'#00CC96','Developing':'#FFA15A','Needs Attention':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    bank_region = df.groupby(['region','bank'])['amount'].sum().reset_index()
    fig = px.bar(bank_region, x='region', y='amount', color='bank', title='🏦 Bank Distribution by Region', barmode='stack')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(regional, x='region', y='failed_rate', color='Performance Tier',
                 title='❌ Failed Transaction Rate by Region (%)',
                 color_discrete_map={'High Performer':'#00CC96','Developing':'#FFA15A','Needs Attention':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📋 Regional Summary Table")
st.dataframe(regional.drop('cluster', axis=1).style.background_gradient(subset=['total_volume','completion_rate'], cmap='RdYlGn'), use_container_width=True)
