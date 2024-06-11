import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
import statistics

def print_mean_std_dev(numbers):
    mean = statistics.mean(numbers)
    std_dev = statistics.stdev(numbers)
    print(f'Mean: {mean}')
    print(f'Standard deviation: {std_dev}')

path = '/Users/elliottunstall/Desktop/Imperial/FYP/Results'
parameters = 'Alpha-0_Beta-1'

# Replace 'filename.csv' with the path to your CSV file
carotid_df = pd.read_csv(f'{path}/soft/10/{parameters}.csv')
cardiac_df = pd.read_csv(f'{path}/cardiac/diastole/{parameters}.csv')

# Extract mean error values for each frame when algorithm is 'dipy_custom'
carotid_errors = carotid_df[carotid_df['Algorithm'] == 'dipy_custom']['Time'].to_list()
carotid_errors_dipy = carotid_df[carotid_df['Algorithm'] == 'dipy']['Time'].to_list()
cardiac_errors = cardiac_df[cardiac_df['Algorithm'] == 'dipy_custom']['Time'].to_list()
cardiac_errors_dipy = cardiac_df[cardiac_df['Algorithm'] == 'dipy']['Time'].to_list()

print_mean_std_dev(cardiac_errors_dipy)
print_mean_std_dev(cardiac_errors)
print_mean_std_dev(carotid_errors_dipy)
print_mean_std_dev(carotid_errors)

# Perform the Wilcoxon signed-rank test
stat_carotid, p_value_carotid = wilcoxon(carotid_errors_dipy, carotid_errors)
stat_cardiac, p_value_cardiac = wilcoxon(cardiac_errors_dipy, cardiac_errors)

print(f'Wilcoxon statistic: {stat_carotid}')
print(f'P-value: {p_value_carotid}')

print(f'Wilcoxon statistic: {stat_cardiac}')
print(f'P-value: {p_value_cardiac}')

import pandas as pd

# Load the dataset
file_path = '/Users/elliottunstall/Desktop/Imperial/FYP/Results/soft/10/all_results.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
data.head()

# Filter the data for algorithm 'dipy' and Beta = 0
filtered_data = data[(data['Algorithm'] == 'dipy') & (data['Beta'] == 0.0)]

# Filter the data for 'SSD' values that are not NaN
filtered_data_ssd = filtered_data['SSD'].dropna()

# Calculate the standard deviation of the SSD values
std_dev_ssd = filtered_data_ssd.std()

print(std_dev_ssd)