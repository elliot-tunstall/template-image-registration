import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = '/Users/elliottunstall/Desktop/Imperial/FYP/Results'
dataset = 'cardiac'
batch = 'diastole'
parameters = 'Alpha-0_Beta-1'

# Assuming 'results_df' is the DataFrame loaded with your data
results_df = pd.read_csv(f'{path}/{dataset}/{batch}/{parameters}.csv')

# Assuming `path` and `parameters` are already defined
# Creating a figure with a subplot for each metric
fig, (axes1, axes2) = plt.subplots(nrows=2, ncols=6, figsize=(12, 10), sharey=False)

for row in range(2):
    axes = axes1
    if row == 1:
        results_df = pd.read_csv(f'{path}/soft/10/{parameters}.csv')
        axes = axes2

    # Melting the DataFrame to make it suitable for seaborn's boxplot
    melted_df = pd.melt(results_df, id_vars=['Frame', 'Algorithm'],
                        value_vars=['SSD', 'MI', 'NCC', 'Mean Error', 'SD Error', 'Time'],
                        var_name='Metric', value_name='Value')

    # Define the unique metrics for subplots
    metrics = melted_df['Metric'].unique()

    # Set context for smaller fonts
    sns.set_context("paper", font_scale=0.8)

    # Define the palette with the correct keys
    palette = {'dipy_custom': 'red', 'dipy': 'blue'}  # Adjust these keys to match your actual algorithms

    # Loop through each metric and create a separate box plot and strip plot
    for ax, metric in zip(axes, metrics):
        subset = melted_df[melted_df['Metric'] == metric]
        
        # Determine the scaling factor
        max_val = subset['Value'].dropna().max()
        exponent = int(np.floor(np.log10(max_val))) if max_val != 0 else 0
        scaling_factor = 10 ** exponent
        
        # Scale the values
        if metric == 'Mean Error' or metric == 'SD Error':
            subset['Scaled Value'] = subset['Value'] * 1000000
        elif exponent > 2 or exponent < -2:
            subset['Scaled Value'] = subset['Value'] / scaling_factor
        else:
            subset['Scaled Value'] = subset['Value']

        # Create the box plot
        boxplot = sns.boxplot(x='Algorithm', y='Scaled Value', data=subset, ax=ax, 
                              palette=palette, whis=[0, 100], dodge=False)  # whis=[0, 100] to include all data points within the whiskers
        
        # Plot mean values as dots
        means = subset.groupby('Algorithm')['Scaled Value'].mean().reset_index()
        for i, algorithm in enumerate(means['Algorithm']):
            mean_value = means.loc[means['Algorithm'] == algorithm, 'Scaled Value'].values[0]
            ax.plot(i, mean_value, 'o', color='black', markersize=6)

        # Add strip plot
        sns.stripplot(x='Algorithm', y='Scaled Value', data=subset, ax=ax, 
                      palette=palette, size=3, jitter=True, edgecolor='gray', alpha=0.7, dodge=False)

        # Update the metric name in the title
        updated_metric = metric.replace('Mean Error', 'Absolute Error').replace('SD Error', 'AE StD').replace('Time', 'Execution Time')
        
        unit = "s" if metric == 'Time' else "AU"  # Replace this with the actual unit if available
        
        if metric == 'Mean Error' or metric == 'SD Error':
            ax.set_title(f'{updated_metric} (μm)', fontsize=11)
        else:
            ax.set_title(f'{updated_metric} ({unit} × 10^{exponent})', fontsize=11) if exponent > 2 or exponent < -2 else ax.set_title(f'{updated_metric} ({unit})', fontsize=11)

        # Remove the frame around each subplot
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes:  # Enable bottom spine for all but the last subplot
        ax.spines['bottom'].set_visible(True)
    for ax in axes:  # Enable top spine for all but the first subplot
        ax.spines['top'].set_visible(True)

# Add a super title to the entire figure
fig.text(0.5, 0.95, 'Cardiac (Systole)', ha='center', fontsize=11, weight='bold')
fig.text(0.5, 0.45, 'Carotid', ha='center', fontsize=11, weight='bold')

# Manually add a legend to the figure
handles = [plt.Line2D([0], [0], color='blue', lw=4),
           plt.Line2D([0], [0], color='red', lw=4)]
labels = ['Baseline', 'Co-Registration']
fig.legend(handles, labels, loc='upper right', title='Algorithm', bbox_to_anchor=(1, 1), ncol=2, fontsize=11)

# Adjust the layout and display the plot
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.90])  # Adjust the tight_layout to consider the legend and increase spacing
plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.4)  # Make space for the super title
plt.show()
