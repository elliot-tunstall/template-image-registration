
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'results_df' is the DataFrame loaded with your data
results_df = pd.read_csv('/Users/elliottunstall/Desktop/Imperial/FYP/Results/soft/10/Alpha-0_Beta-1.csv')

# Creating a box plot for Mean Error across different algorithms
plt.figure(figsize=(10, 6))
sns.boxplot(x='Algorithm', y='Mean Error', data=results_df)
plt.title('Comparison of Mean Error Across Different Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Mean Error')
plt.show()


## Plot 2: Box plot for multiple metrics across different algorithms

# Melting the DataFrame to make it suitable for seaborn's boxplot
melted_df = pd.melt(results_df, id_vars=['Frame', 'Algorithm', 'Parameters'],
                    value_vars=['pSSD', 'Mean Error', 'SD Error', 'Time'],
                    var_name='Metric', value_name='Value')

# Define the unique metrics for subplots
metrics = melted_df['Metric'].unique()

# Creating a figure with a subplot for each metric
# Adjust the figure size if needed to accommodate all elements
fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(20, 6), sharey=False)

# Loop through each metric and create a separate box plot and strip plot
for ax, metric in zip(axes, metrics):
    # Boxplot with legend, will use these handles for the final legend
    sns.boxplot(x='Metric', y='Value', hue='Algorithm', data=melted_df[melted_df['Metric'] == metric], ax=ax, 
                palette='Set2', fliersize=0)  # Hide outliers since they will be shown in stripplot
    # Stripplot with colors matching the hue
    sns.stripplot(x='Metric', y='Value', hue='Algorithm', data=melted_df[melted_df['Metric'] == metric], ax=ax, 
                  palette='Set2', size=3, jitter=True, dodge=True, edgecolor='gray', alpha=0.7, legend=False)
    ax.set_title(metric)
    ax.set_xlabel('')
    ax.set_ylabel('Value')
    if ax is not axes[0]:  # Remove the y-axis label for subplots other than the first for clarity
        ax.set_ylabel('')
    # Hide boxplot's automatic legend, we will create a manual one later
    ax.get_legend().remove()

# Manually add a legend from the last boxplot to the figure
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', title='Algorithm')

# Adjust the layout and display the plot
plt.tight_layout()  # Adjust the tight_layout to consider the legend
plt.show()


## Plot 3: Line plot of error over frames

# Setup the figure and subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=False)

# Filter data by algorithms if needed
algorithms = results_df['Algorithm'].unique()  # Unique algorithms

colors = ['tab:red', 'tab:blue']  # Color for each algorithm for clarity

# First Subplot: Mean Error with Std Deviation
for alg, color in zip(algorithms, colors):
    alg_data = results_df[results_df['Algorithm'] == alg]
    ax1.errorbar(alg_data['Frame'], alg_data['Mean Error'], yerr=alg_data['SD Error'], 
                 label=f'{alg}', fmt='-o', color=color)
ax1.set_title('Comparison of Mean Error by Algorithm')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Mean Error')
ax1.legend()

# Second Subplot: pSSD against Frame
for alg, color in zip(algorithms, colors):
    alg_data = results_df[results_df['Algorithm'] == alg]
    ax2.plot(alg_data['Frame'], alg_data['pSSD'], label=f'{alg}', marker='o', color=color)
ax2.plot(alg_data['Frame'], alg_data['DICE'], label=f'Dice Score', marker='x', color='tab:green')
ax2.set_title('Comparison of pSSD by Algorithm')
ax2.set_xlabel('Frame')
ax2.set_ylabel('pSSD')
ax2.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()