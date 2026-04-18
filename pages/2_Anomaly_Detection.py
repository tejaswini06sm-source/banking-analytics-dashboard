import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection — Suspicious Transaction Flagging")
st.markdown("ML-powered anomaly detection using Isolation Forest to identify suspicious transactions.")
st.markdown("---")

df = load_data()

st.sidebar.title("⚙️ Model Settings")
contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.15, 0.05, 0.01,
                                   help="Expected proportion of anomalies in data")
selected_banks = st.sidebar.multiselect("Filter Bank", df['bank'].unique(), default=list(df['bank'].unique()))
df = df[df['bank'].isin(selected_banks)]

scaler = StandardScaler()
features = scaler.fit_transform(df[['amount', 'processing_time_hrs']])
model = IsolationForest(contamination=contamination, random_state=42)
df['anomaly'] = model.fit_predict(features)
df['anomaly_score'] = model.score_samples(features)
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: '🔴 Anomaly'})

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 Anomalies Detected", f"{len(df[df['anomaly']==-1]):,}")
col2.metric("Anomaly Rate", f"{(df['anomaly']==-1).mean()*100:.1f}%")
col3.metric("Avg Anomalous Amount", f"₹{df[df['anomaly']==-1]['amount'].mean():,.0f}")
col4.metric("Max Suspicious Amount", f"₹{df[df['anomaly']==-1]['amount'].max():,.0f}")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(df, x='amount', y='processing_time_hrs', color='anomaly_label',
                     color_discrete_map={'Normal':'#636EFA','🔴 Anomaly':'#EF553B'},
                     title='🔍 Anomaly Scatter: Amount vs Processing Time',
                     hover_data=['transaction_id','bank','status'], opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x='anomaly_score', color='anomaly_label', nbins=50,
                       title='📊 Anomaly Score Distribution',
                       color_discrete_map={'Normal':'#636EFA','🔴 Anomaly':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    anomaly_bank = df.groupby(['bank','anomaly_label'])['transaction_id'].count().reset_index()
    fig = px.bar(anomaly_bank, x='bank', y='transaction_id', color='anomaly_label',
                 title='🏦 Anomalies by Bank', barmode='group',
                 color_discrete_map={'Normal':'#636EFA','🔴 Anomaly':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    anomaly_type = df.groupby(['transaction_type','anomaly_label'])['transaction_id'].count().reset_index()
    fig = px.bar(anomaly_type, x='transaction_type', y='transaction_id', color='anomaly_label',
                 title='💳 Anomalies by Transaction Type', barmode='group',
                 color_discrete_map={'Normal':'#636EFA','🔴 Anomaly':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    monthly_anomaly = df[df['anomaly']==-1].groupby(df['date'].dt.to_period('M').astype(str))['transaction_id'].count().reset_index()
    fig = px.line(monthly_anomaly, x='date', y='transaction_id', title='📈 Anomaly Trend Over Time',
                  markers=True, color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(df, x='anomaly_label', y='amount', color='anomaly_label',
                 title='💰 Amount Range: Normal vs Anomaly',
                 color_discrete_map={'Normal':'#636EFA','🔴 Anomaly':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔴 Flagged Transactions — Full List")
flagged = df[df['anomaly']==-1][['transaction_id','date','bank','region','transaction_type','amount','status','processing_time_hrs','anomaly_score']].sort_values('amount', ascending=False)
st.dataframe(flagged, use_container_width=True)
