import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Transaction Analysis", page_icon="📊", layout="wide")
st.title("📊 Transaction Analysis")

df = pd.read_csv('data/transactions.csv', parse_dates=['date'])

# Filters
col1, col2 = st.columns(2)
with col1:
    banks = st.multiselect("Filter by Bank", df['bank'].unique(), default=df['bank'].unique())
with col2:
    types = st.multiselect("Filter by Type", df['transaction_type'].unique(), default=df['transaction_type'].unique())

df = df[df['bank'].isin(banks) & df['transaction_type'].isin(types)]

# Charts
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