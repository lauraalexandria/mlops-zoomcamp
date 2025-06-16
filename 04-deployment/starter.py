#!/usr/bin/env python
# coding: utf-8

import argparse


# In[3]:


import pickle
import pandas as pd


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:

parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
args = parser.parse_args()

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet')


# In[9]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[11]:


print(y_pred.mean())


# In[13]:

df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')


# In[15]:


df_result = pd.DataFrame({'ride_id': df['ride_id'], 'prediction': y_pred})


# In[16]:


df_result.to_parquet(
    "final_data",
    engine='pyarrow',
    compression=None,
    index=False
)

