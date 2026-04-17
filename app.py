import streamlit as st
import pandas as pd

st.set_page_config(page_title="Banking Operations Analytics", page_icon="🏦", layout="wide")

st.title("🏦 Banking Operations Analytics Dashboard")
st.markdown("#### Operational Intelligence Platform — Transaction Monitoring & Risk Reporting")
st.markdown("---")

df = pd.read_csv('data/transactions.csv', parse_dates=['date'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Total Volume", f"₹{df['amount'].sum()/1e7:.2f} Cr")
col3.metric("Completion Rate", f"{(df['status']=='Completed').mean()*100:.1f}%")
col4.metric("Avg Processing Time", f"{df['processing_time_hrs'].mean():.2f} hrs")

st.markdown("---")
st.markdown("### 📊 Use the sidebar to navigate to detailed analysis pages")
st.dataframe(df.head(20), use_container_width=True)