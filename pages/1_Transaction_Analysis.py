import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
        'date': dates,
        'bank': np.random.choice(banks, n),
        'region': np.random.choice(regions, n),
        'transaction_type': np.random.choice(transaction_types, n),
        'amount': np.round(amounts, 2),
        'status': np.random.choice(statuses, n),
        'processing_time_hrs': np.round(np.random.exponential(2, n), 2)
    })

st.set_page_config(page_title="Transaction Analysis", page_icon="📊", layout="wide")
st.title("📊 Transaction Analysis")

df = load_data()

col1, col2 = st.columns(2)
with col1:
    banks = st.multiselect("Filter by Bank", df['bank'].unique(), default=list(df['bank'].unique()))
with col2:
    types = st.multiselect("Filter by Type", df['transaction_type'].unique(), default=list(df['transaction_type'].unique()))

df = df[df['bank'].isin(banks) & df['transaction_type'].isin(types)]

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df.groupby('bank')['amount'].sum().reset_index(), x='bank', y='amount', title='Total Volume by Bank', color='bank')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.pie(df, names='transaction_type', title='Transaction Type Distribution')
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df.groupby('status')['transaction_id'].count().reset_index(), x='status', y='transaction_id', title='Transaction Status Breakdown', color='status')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    monthly = df.groupby(df['date'].dt.to_period('M').astype(str))['amount'].sum().reset_index()
    fig = px.line(monthly, x='date', y='amount', title='Monthly Transaction Volume Trend')
    st.plotly_chart(fig, use_container_width=True)
