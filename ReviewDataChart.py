# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:01:14 2024

@author: GULO1L
"""
import DataPrep
import matplotlib.pyplot as plt
import seaborn as sns

# Sample a few pipelines for visualization of rolling metrics (to keep it clear)
sample_pipelines = DataPrep.df_realistic['pipeline_name'].unique()[:5]  # Use the first 5 pipelines as samples

# Plot 1: Failure Rate by Pipeline
plt.figure(figsize=(12, 6))
sns.barplot(data=DataPrep.df_realistic.drop_duplicates('pipeline_name'), x='pipeline_name', y='failure_rate')
plt.xticks(rotation=90)
plt.title("Failure Rate by Pipeline")
plt.xlabel("Pipeline Name")
plt.ylabel("Failure Rate")
plt.show()

# Plot 2: Average Run Duration vs. Failure Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(data=DataPrep.df_realistic.drop_duplicates('pipeline_name'), x='avg_run_duration', y='failure_rate')
plt.title("Average Run Duration vs. Failure Rate")
plt.xlabel("Average Run Duration (s)")
plt.ylabel("Failure Rate")
plt.show()

# Plot 3: Run Duration Variance by Pipeline
plt.figure(figsize=(12, 6))
sns.barplot(data=DataPrep.df_realistic.drop_duplicates('pipeline_name'), x='pipeline_name', y='run_duration_variance')
plt.xticks(rotation=90)
plt.title("Run Duration Variance by Pipeline")
plt.xlabel("Pipeline Name")
plt.ylabel("Run Duration Variance")
plt.show()

# Plot 4: Rolling Average Run Time for Sample Pipelines
plt.figure(figsize=(12, 6))
for pipeline in sample_pipelines:
    subset = DataPrep.df_realistic[DataPrep.df_realistic['pipeline_name'] == pipeline]
    plt.plot(subset['start_time'], subset['rolling_avg_run_time'], label=pipeline)
plt.title("Rolling Average Run Time for Sample Pipelines")
plt.xlabel("Start Time")
plt.ylabel("Rolling Average Run Time (s)")
plt.legend()
plt.show()

# Plot 5: Rolling Failure Rate for Sample Pipelines
plt.figure(figsize=(12, 6))
for pipeline in sample_pipelines:
    subset = DataPrep.df_realistic[DataPrep.df_realistic['pipeline_name'] == pipeline]
    plt.plot(subset['start_time'], subset['rolling_failure_rate'], label=pipeline)
plt.title("Rolling Failure Rate for Sample Pipelines")
plt.xlabel("Start Time")
plt.ylabel("Rolling Failure Rate")
plt.legend()
plt.show()


# Plot 6 : Calculating success rate by month
DataPrep.df_realistic['month'] = DataPrep.df_realistic['start_time'].dt.to_period("M")
success_rate = DataPrep.df_realistic.groupby(['pipeline_name', 'month'])['fail_flag'].apply(lambda x: 1 - x.mean()).reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=success_rate, x='month', y='fail_flag', hue='pipeline_name', marker="o")
plt.title("Pipeline Success Rate Over Time")
plt.xlabel("Month")
plt.ylabel("Success Rate")
plt.legend(title='Pipeline', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

