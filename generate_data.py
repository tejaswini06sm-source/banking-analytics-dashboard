import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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

df = pd.DataFrame({
    'transaction_id': [f'TXN{str(i).zfill(5)}' for i in range(n)],
    'date': dates,
    'bank': np.random.choice(banks, n),
    'region': np.random.choice(regions, n),
    'transaction_type': np.random.choice(transaction_types, n),
    'amount': np.round(amounts, 2),
    'status': np.random.choice(statuses, n),
    'processing_time_hrs': np.round(np.random.exponential(2, n), 2)
})

df.to_csv('data/transactions.csv', index=False)
print("Dataset created: data/transactions.csv")