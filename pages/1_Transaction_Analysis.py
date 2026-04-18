import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

st.set_page_config(page_title="Transaction Analysis", page_icon="📊", layout="wide")
st.title("📊 Transaction Analysis")
st.markdown("Deep dive into transaction patterns, volumes, and trends across banks and payment types.")
st.markdown("---")

df = load_data()

st.sidebar.title("🔎 Filters")
banks = st.sidebar.multiselect("Bank", df['bank'].unique(), default=list(df['bank'].unique()))
types = st.sidebar.multiselect("Transaction Type", df['transaction_type'].unique(), default=list(df['transaction_type'].unique()))
regions = st.sidebar.multiselect("Region", df['region'].unique(), default=list(df['region'].unique()))
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])
df = df[df['bank'].isin(banks) & df['transaction_type'].isin(types) & df['region'].isin(regions)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Transactions", f"{len(df):,}")
col2.metric("Total Volume", f"₹{df['amount'].sum()/1e7:.2f} Cr")
col3.metric("Avg Amount", f"₹{df['amount'].mean():,.0f}")
col4.metric("Completion Rate", f"{(df['status']=='Completed').mean()*100:.1f}%")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df.groupby('bank')['amount'].sum().reset_index().sort_values('amount', ascending=False),
                 x='bank', y='amount', color='bank', title='💰 Total Volume by Bank', text_auto='.2s')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    monthly = df.groupby(df['date'].dt.to_period('M').astype(str))['amount'].sum().reset_index()
    fig = px.area(monthly, x='date', y='amount', title='📈 Monthly Volume Trend', color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.pie(df, names='transaction_type', title='🔄 Transaction Type Distribution', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.pie(df, names='status', title='✅ Status Breakdown', hole=0.4,
                 color_discrete_map={'Completed':'#00CC96','Pending':'#FFA15A','Failed':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    heatmap = df.groupby(['bank','transaction_type'])['amount'].sum().reset_index()
    heatmap_pivot = heatmap.pivot(index='bank', columns='transaction_type', values='amount').fillna(0)
    fig = px.imshow(heatmap_pivot, title='🔥 Volume Heatmap: Bank vs Transaction Type', color_continuous_scale='Blues', text_auto='.2s')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(df, x='bank', y='amount', color='bank', title='📦 Amount Distribution by Bank')
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    hourly = df.groupby(df['date'].dt.day_of_week)['transaction_id'].count().reset_index()
    hourly.columns = ['day_of_week', 'count']
    hourly['day'] = hourly['day_of_week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
    fig = px.bar(hourly, x='day', y='count', title='📅 Transactions by Day of Week', color='count', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(df.groupby(['region','status'])['transaction_id'].count().reset_index(),
                 x='region', y='transaction_id', color='status', title='🗺️ Status by Region', barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📋 Filtered Transaction Data")
st.dataframe(df.sort_values('amount', ascending=False).head(50), use_container_width=True)
