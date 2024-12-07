# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:54:57 2024

@author: GULO1L
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('pipeline_log.csv')

# Feature engineering
# Calculate the total number of runs for each pipeline
pipeline_run_counts = data.groupby('pipeline_name').size().reset_index(name='run_count')

# Merge the run count back to the original data
data = pd.merge(data, pipeline_run_counts, on='pipeline_name')

# Filter out pipelines that have run fewer than 5 times
data = data[data['run_count'] >= 5]

# Preprocessing: Create features from the start and end times (optional)
data['start_time'] = pd.to_datetime(data['start_time'])
data['end_time'] = pd.to_datetime(data['end_time'])

# Calculate duration in seconds for each run (this could be a feature)
data['duration'] = (data['end_time'] - data['start_time']).dt.total_seconds()

# Copy data frame
df_realistic = data 

# Assuming df_realistic is your existing DataFrame
df_realistic.sort_values(by=['pipeline_name', 'start_time'], inplace=True)

# Adding Hour and Day of Week features
df_realistic['hour_of_day'] = df_realistic['start_time'].dt.hour
df_realistic['day_of_week'] = df_realistic['start_time'].dt.dayofweek
df_realistic['is_weekend'] = df_realistic['day_of_week'] >= 5

# Calculating cumulative run count per pipeline
df_realistic['run_count'] = df_realistic.groupby('pipeline_name').cumcount() + 1

# Previous Run Failure feature
df_realistic['previous_fail_flag'] = df_realistic.groupby('pipeline_name')['fail_flag'].shift(1).fillna(0)

# Time Since Last Run
df_realistic['time_since_last_run'] = df_realistic.groupby('pipeline_name')['start_time'].diff().dt.total_seconds().fillna(0)

# Calculating failure rate for each pipeline
failure_rate = df_realistic.groupby('pipeline_name')['fail_flag'].transform('mean')
df_realistic['failure_rate'] = failure_rate

# Average Run Duration and Variance per pipeline
df_realistic['avg_run_duration'] = df_realistic.groupby('pipeline_name')['run_time'].transform('mean')
df_realistic['run_duration_variance'] = df_realistic.groupby('pipeline_name')['run_time'].transform('var').fillna(0)

# Rolling Metrics: Rolling average run time and rolling failure rate for last 5 runs
df_realistic['rolling_avg_run_time'] = df_realistic.groupby('pipeline_name')['run_time'].transform(lambda x: x.rolling(5, 1).mean())
df_realistic['rolling_failure_rate'] = df_realistic.groupby('pipeline_name')['fail_flag'].transform(lambda x: x.rolling(5, 1).mean())

# Display the DataFrame with new features
import ace_tools as tools; tools.display_dataframe_to_user(name="Pipeline Execution Logs with New Features", dataframe=df_realistic)
