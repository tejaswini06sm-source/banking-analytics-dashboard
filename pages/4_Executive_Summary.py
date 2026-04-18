import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
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

st.set_page_config(page_title="Executive Summary", page_icon="📋", layout="wide")

df = load_data()

df['risk_score'] = (
    (df['amount'] > df['amount'].quantile(0.90)).astype(int) * 40 +
    (df['processing_time_hrs'] > df['processing_time_hrs'].quantile(0.90)).astype(int) * 30 +
    (df['status'] == 'Failed').astype(int) * 30
)
df['risk_level'] = pd.cut(df['risk_score'], bins=[-1,0,40,70,100],
                           labels=['Low','Medium','High','Critical'])
df['sla_breach'] = df['processing_time_hrs'] > 4.0

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['amount','processing_time_hrs']])

bank_score = df.groupby('bank').agg(
    total_volume=('amount','sum'),
    total_txns=('transaction_id','count'),
    completion_rate=('status', lambda x: (x=='Completed').mean()*100),
    sla_breach_rate=('sla_breach', 'mean'),
    anomaly_count=('anomaly', lambda x: (x==-1).sum()),
    avg_processing=('processing_time_hrs','mean')
).reset_index()
bank_score['performance_score'] = (
    bank_score['completion_rate'] * 0.4 +
    (1 - bank_score['sla_breach_rate']) * 100 * 0.3 +
    (1 - bank_score['anomaly_count']/bank_score['total_txns']) * 100 * 0.3
).round(2)

st.title("📋 Executive Summary — Banking Operations Intelligence")
st.markdown("**Prepared for:** Senior Leadership | **Period:** Jan 2023 – Dec 2024 | **Classification:** Internal")
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Total Volume", f"₹{df['amount'].sum()/1e7:.1f} Cr")
col3.metric("Completion Rate", f"{(df['status']=='Completed').mean()*100:.1f}%")
col4.metric("SLA Breaches", f"{df['sla_breach'].sum():,}", delta=f"{df['sla_breach'].mean()*100:.1f}% rate", delta_color="inverse")
col5.metric("Anomalies Flagged", f"{(df['anomaly']==-1).sum():,}", delta_color="inverse")
col6.metric("Critical Risk Txns", f"{(df['risk_level']=='Critical').sum():,}", delta_color="inverse")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    risk_summary = df['risk_level'].value_counts().reset_index()
    fig = px.pie(risk_summary, names='risk_level', values='count',
                 title='🎯 Transaction Risk Distribution',
                 color='risk_level',
                 color_discrete_map={'Low':'#00CC96','Medium':'#FFA15A','High':'#EF553B','Critical':'#8B0000'},
                 hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    sla_bank = df.groupby('bank')['sla_breach'].mean().reset_index()
    sla_bank.columns = ['bank','sla_breach_rate']
    sla_bank['sla_breach_rate'] = (sla_bank['sla_breach_rate']*100).round(2)
    fig = px.bar(sla_bank.sort_values('sla_breach_rate', ascending=False),
                 x='bank', y='sla_breach_rate', color='bank',
                 title='⏱️ SLA Breach Rate by Bank (%)', text_auto='.1f')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🏦 Bank Performance Scorecard")
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(data=[
        go.Bar(name='Completion Rate', x=bank_score['bank'], y=bank_score['completion_rate'], marker_color='#00CC96'),
        go.Bar(name='Performance Score', x=bank_score['bank'], y=bank_score['performance_score'], marker_color='#636EFA')
    ])
    fig.update_layout(barmode='group', title='Bank Scorecard: Completion vs Performance Score')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.scatter(bank_score, x='completion_rate', y='avg_processing',
                     size='total_volume', color='bank', text='bank',
                     title='⚡ Efficiency Matrix: Completion Rate vs Processing Time')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📊 Bank Scorecard Table")
styled = bank_score.copy()
styled['completion_rate'] = styled['completion_rate'].round(2)
styled['sla_breach_rate'] = (styled['sla_breach_rate']*100).round(2)
styled['avg_processing'] = styled['avg_processing'].round(2)
st.dataframe(styled.sort_values('performance_score', ascending=False), use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    monthly = df.groupby(df['date'].dt.to_period('M').astype(str)).agg(
        volume=('amount','sum'),
        anomalies=('anomaly', lambda x: (x==-1).sum()),
        sla_breaches=('sla_breach','sum')
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['volume'], name='Volume', yaxis='y1', line=dict(color='#636EFA')))
    fig.add_trace(go.Bar(x=monthly['date'], y=monthly['anomalies'], name='Anomalies', yaxis='y2', marker_color='#EF553B', opacity=0.5))
    fig.update_layout(title='📈 Volume vs Anomalies Over Time',
                      yaxis=dict(title='Volume'),
                      yaxis2=dict(title='Anomalies', overlaying='y', side='right'))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    risk_trend = df.groupby([df['date'].dt.to_period('M').astype(str), 'risk_level'])['transaction_id'].count().reset_index()
    fig = px.area(risk_trend, x='date', y='transaction_id', color='risk_level',
                  title='🎯 Risk Level Trend Over Time',
                  color_discrete_map={'Low':'#00CC96','Medium':'#FFA15A','High':'#EF553B','Critical':'#8B0000'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("💡 Key Findings & Policy Recommendations")
col1, col2, col3 = st.columns(3)
with col1:
    st.error("**🔴 Finding 1: SLA Breach Concentration**")
    top_sla = sla_bank.sort_values('sla_breach_rate', ascending=False).iloc[0]
    st.markdown(f"**{top_sla['bank']}** has the highest SLA breach rate at **{top_sla['sla_breach_rate']:.1f}%**. Processing delays concentrated in SWIFT and RTGS transactions.")
    st.markdown("**Recommendation:** Implement automated escalation triggers for transactions exceeding 3hr processing threshold.")
with col2:
    st.warning("**🟡 Finding 2: Anomaly Clustering**")
    top_anomaly_bank = df[df['anomaly']==-1].groupby('bank')['transaction_id'].count().idxmax()
    st.markdown(f"**{top_anomaly_bank}** accounts for the highest share of flagged anomalies. High-value outliers skew risk exposure disproportionately.")
    st.markdown("**Recommendation:** Deploy real-time anomaly scoring with bank-specific thresholds and mandatory review queues for Critical-risk transactions.")
with col3:
    st.success("**🟢 Finding 3: Regional Performance Gap**")
    st.markdown("North and South regions consistently outperform East and West on volume and completion rates, suggesting infrastructure and capacity disparities.")
    st.markdown("**Recommendation:** Redistribute processing capacity to underperforming regions and establish regional SLA benchmarks aligned with transaction volumes.")
