import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

st.set_page_config(page_title="Banking Operations Analytics", page_icon="🏦", layout="wide")

# Generate data inline — no external file needed
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

df = load_data()

st.title("🏦 Banking Operations Analytics Dashboard")
st.markdown("#### Operational Intelligence Platform — Transaction Monitoring & Risk Reporting")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Total Volume", f"₹{df['amount'].sum()/1e7:.2f} Cr")
col3.metric("Completion Rate", f"{(df['status']=='Completed').mean()*100:.1f}%")
col4.metric("Avg Processing Time", f"{df['processing_time_hrs'].mean():.2f} hrs")

st.markdown("---")
st.markdown("### 📊 Use the sidebar to navigate to detailed analysis pages")
st.dataframe(df.head(20), use_container_width=True)
