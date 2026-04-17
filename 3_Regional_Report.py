import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Regional Report", page_icon="🗺️", layout="wide")
st.title("🗺️ Regional Performance Report")

df = pd.read_csv('data/transactions.csv', parse_dates=['date'])

regional = df.groupby('region').agg(
    total_transactions=('transaction_id','count'),
    total_volume=('amount','sum'),
    avg_amount=('amount','mean'),
    completion_rate=('status', lambda x: (x=='Completed').mean()*100),
    avg_processing=('processing_time_hrs','mean')
).reset_index()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
regional['performance_tier'] = kmeans.fit_predict(regional[['total_volume','completion_rate']])
regional['performance_tier'] = regional['performance_tier'].map({0:'Needs Attention', 1:'High Performer', 2:'Developing'})

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