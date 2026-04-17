import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection — Suspicious Transaction Flagging")

df = pd.read_csv('data/transactions.csv', parse_dates=['date'])

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

st.subheader("🔴 Flagged Transactions")
st.dataframe(df[df['anomaly']==-1][['transaction_id','date','bank','amount','status','processing_time_hrs']].sort_values('amount', ascending=False), use_container_width=True)