import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
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

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection — Suspicious Transaction Flagging")

df = load_data()

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['amount', 'processing_time_hrs']])
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

col1, col2, col3 = st.columns(3)
col1.metric("Total Anomalies Detected", len(df[df['anomaly']==-1]))
col2.metric("Anomaly Rate", f"{(df['anomaly']==-1).mean()*100:.1f}%")
col3.metric("Avg Anomalous Amount", f"₹{df[df['anomaly']==-1]['amount'].mean():,.0f}")

fig = px.scatter(df, x='amount', y='processing_time_hrs', color='anomaly_label',
                 color_discrete_map={'Normal':'#636EFA','Anomaly':'#EF553B'},
                 title='Anomaly Detection — Amount vs Processing Time',
                 hover_data=['transaction_id','bank','status'])
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df.groupby(['bank','anomaly_label'])['transaction_id'].count().reset_index(),
                 x='bank', y='transaction_id', color='anomaly_label', title='Anomalies by Bank', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(df.groupby(['transaction_type','anomaly_label'])['transaction_id'].count().reset_index(),
                 x='transaction_type', y='transaction_id', color='anomaly_label', title='Anomalies by Transaction Type', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔴 Flagged Transactions")
st.dataframe(df[df['anomaly']==-1][['transaction_id','date','bank','amount','status','processing_time_hrs']].sort_values('amount', ascending=False), use_container_width=True)
