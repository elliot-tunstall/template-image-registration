
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
import glob
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Setting default parameters
mpl.rcParams['axes.labelsize'] = 11  # fontsize for x-axis and y-axis labels
mpl.rcParams['xtick.labelsize'] = 8  # fontsize for x-axis tick labels
mpl.rcParams['ytick.labelsize'] = 8  # fontsize for y-axis tick labels
mpl.rcParams['figure.titlesize'] = 12  # fontsize for the overall plot title
mpl.rcParams['figure.titleweight'] = 'bold'  # fontweight for the overall plot title
mpl.rcParams['lines.linewidth'] = 2  # default line width

# Define a custom color palette using seaborn
husl = sns.color_palette(None, 3)  # Generates a palette of 3 distinct soft colors

# Formatter function to use scientific notation
def scientific_formatter(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)

path = '/Users/elliottunstall/Desktop/Imperial/FYP/Results'
dataset = 'cardiac'
batch = 'diastole'
parameters = 'Alpha-0_Beta-1'

# Assuming 'results_df' is the DataFrame loaded with your data
results_df = pd.read_csv(f'{path}/{dataset}/{batch}/{parameters}.csv')

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
                    value_vars=['SSD', 'MI', 'NCC', 'Mean Error', 'SD Error', 'Time'],
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
true_mask_df = pd.read_csv(f'{path}/{dataset}/{batch}/true_segmentation/{parameters}.csv')

# Setup the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=False)

# Filter data by algorithms if needed
algorithms = results_df['Algorithm'].unique()  # Unique algorithms

colors = ['tab:red', 'tab:blue']  # Color for each algorithm for clarity

# First Subplot: Mean Error with Std Deviation
# Plot data with fill_between for each algorithm
for alg, color in zip(algorithms, colors):
    alg_data = results_df[results_df['Algorithm'] == alg]
    frame = alg_data['Frame']
    mean_error = alg_data['Mean Error']
    sd_error = alg_data['SD Error']
    lower_bound = mean_error - sd_error
    upper_bound = mean_error + sd_error
    
    ax1.plot(frame, mean_error, label=f'{alg}', color=color)
    ax1.fill_between(frame, lower_bound, upper_bound, color=color, alpha=0.2)

# Plot data with fill_between for the true mask
true_mask_data = true_mask_df[true_mask_df['Algorithm'] == 'dipy_custom']
frame_true = true_mask_data['Frame']
mean_error_true = true_mask_data['Mean Error']
sd_error_true = true_mask_data['SD Error']
lower_bound_true = mean_error_true - sd_error_true
upper_bound_true = mean_error_true + sd_error_true

ax1.plot(frame_true, mean_error_true, label='Co-Registration (True Mask)', color='tab:orange')
ax1.fill_between(frame_true, lower_bound_true, upper_bound_true, color='tab:orange', alpha=0.2)

# Optionally, plot DICE score (commented out in original code)
# ax1.plot(alg_data['Frame'], (1 - alg_data['DICE']) / 100, label='Dice Score', marker='x', color='tab:green')

ax1.set_title('Comparison of Mean Error by Algorithm')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Absolute Error')
ax1.legend()

# Second Subplot: Mean Error with Std Deviation
# Extract x and y values
dipy_data = results_df[results_df['Algorithm'] == 'dipy']
coreg_data = results_df[results_df['Algorithm'] == 'dipy_custom']

x = 1 - dipy_data['DICE']
y = dipy_data['Mean Error']
y2 = coreg_data['Mean Error']

# Fit a linear model (line of best fit)
p = np.polyfit(x, y, 1)  # Linear fit (degree 1 polynomial)
y_fit = np.polyval(p, x)  # Evaluate the polynomial at x

p2 = np.polyfit(x, y2, 1)  # Linear fit (degree 1 polynomial)
y2_fit = np.polyval(p2, x)  # Evaluate the polynomial at x

# Scatter plot
ax2.scatter(x, y, color='tab:blue')
ax2.scatter(x, y2, color='tab:red')

# Best fit line
ax2.plot(x, y_fit, label='Dipy', color='tab:blue')
ax2.plot(x, y2_fit, label='Co-Registration', color='tab:red')

# Add error bars
ax2.errorbar(x, y, yerr=dipy_data['SD Error'], fmt='o', color='tab:blue', alpha=0.5)
ax2.errorbar(x, y2, yerr=coreg_data['SD Error'], fmt='o', color='tab:red', alpha=0.5)

ax2.set_title('Comparison of Mean Error by Algorithm')
ax2.set_xlabel('1 - DICE Score')
ax2.set_ylabel('Absolute Error')
ax2.legend()

# Third Subplot: pSSD against Frame

# Read data
cardiac_df = pd.read_csv(f'{path}/cardiac/diastole/{parameters}.csv')
carotid_df = pd.read_csv(f'{path}/soft/10/{parameters}.csv')

cardiac_data = cardiac_df[cardiac_df['Algorithm'] == 'dipy_custom']
carotid_data = carotid_df[carotid_df['Algorithm'] == 'dipy_custom']

# Extract x and y values
x_cardiac = 1 - cardiac_data['DICE']
x_carotid = 1 - carotid_data['DICE']
y_cardiac = cardiac_data['Mean Error']
y_carotid = carotid_data['Mean Error']
sd_error_cardiac = cardiac_data['SD Error']
sd_error_carotid = carotid_data['SD Error']

# Ensure there are no NaN values
if x_cardiac.isnull().any() or y_cardiac.isnull().any() or x_carotid.isnull().any() or y_carotid.isnull().any():
    raise ValueError("NaN values found in the data. Check the data preparation steps.")

# Fit a linear regression model
x_cardiac_with_intercept = sm.add_constant(x_cardiac)
x_carotid_with_intercept = sm.add_constant(x_carotid)
model1 = OLS(y_cardiac, x_cardiac_with_intercept).fit()
model2 = OLS(y_carotid, x_carotid_with_intercept).fit()

# Predict the regression line
y_cardiac_fit = model1.predict(x_cardiac_with_intercept)
y_carotid_fit = model2.predict(x_carotid_with_intercept)

# Get the confidence intervals
ci1 = model1.conf_int(alpha=0.05)
ci2 = model2.conf_int(alpha=0.05)

# Add R^2 value to the plot
r2_cardiac = model1.rsquared
r2_carotid = model2.rsquared

# Get the confidence intervals
pred_cardiac = model1.get_prediction(x_cardiac_with_intercept).summary_frame(alpha=0.05)
pred_carotid = model2.get_prediction(x_carotid_with_intercept).summary_frame(alpha=0.05)

# Create a scatter plot
ax3.scatter(x_cardiac, y_cardiac, color='tab:blue', label='Cardiac')
ax3.scatter(x_carotid, y_carotid, color='tab:red', label='Carotid')

# Plot the regression lines
ax3.plot(x_cardiac, y_cardiac_fit, color='tab:blue', label=f'Cardiac ($R^2$ = {r2_cardiac:.2f})')
ax3.plot(x_carotid, y_carotid_fit, color='tab:red', label=f'Carotid ($R^2$ = {r2_carotid:.2f})')

# Add error bars
ax3.errorbar(x_cardiac, y_cardiac, yerr=sd_error_cardiac, fmt='o', color='tab:blue', alpha=0.5)
ax3.errorbar(x_carotid, y_carotid, yerr=sd_error_carotid, fmt='o', color='tab:red', alpha=0.5)

# Plot confidence intervals
ax3.fill_between(x_cardiac, pred_cardiac['mean_ci_lower'], pred_cardiac['mean_ci_upper'], color='blue', alpha=0.2)
ax3.fill_between(x_carotid, pred_carotid['mean_ci_lower'], pred_carotid['mean_ci_upper'], color='red', alpha=0.2)

# Add titles and labels
ax3.set_title('Regression of Absolute Error on DICE Score')
ax3.set_xlabel('1 - DICE Score')
ax3.set_ylabel('Absolute Error')
ax3.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

## Plot 4: Box plot for multiple metrics across different algorithms with manual legend

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
                            palette=palette, whis=[0, 100], dodge=False)  # Set whis=[0, 100] to include all data points within the whiskers
        for patch in boxplot.patches:
            patch.set_alpha(0.7)  # Set the transparency level here

        sns.stripplot(x='Algorithm', y='Scaled Value', data=subset, ax=ax, 
                    palette=palette, size=3, jitter=True, edgecolor='gray', alpha=0.7, dodge=False, legend=False)
        # Update the metric name in the title
        updated_metric = metric.replace('Mean Error', 'Absolute Error').replace('SD Error', 'AE StD').replace('Time', 'Execution Time')
        # ax.set_title(updated_metric, fontsize=9)
        ax.set_xlabel('')  # Remove x-axis title
        ax.set_ylabel('')  # Remove y-axis title
        ax.set_xticklabels([''] * len(subset['Algorithm'].unique()))  # Remove the x-axis tick labels
        if metric == 'Time':
            unit = "s"
        else:
            unit = "AU"  # Replace this with the actual unit if available
        if metric == 'Mean Error' or metric == 'SD Error':
            # ax.set_ylabel(f'{updated_metric} (\u00B5m)')
            ax.set_title(f'{updated_metric} (\u00B5m)', fontsize=11)
        else:
            # ax.set_ylabel(f'{updated_metric} ({unit} $\\times 10^{exponent}$)') if exponent > 2 or exponent < -2 else ax.set_ylabel(f'{updated_metric} ({unit})')
            ax.set_title(f'{updated_metric} ({unit} $\\times 10^{exponent}$)', fontsize=11) if exponent > 2 or exponent < -2 else ax.set_title(f'{updated_metric} ({unit})', fontsize=11)
            if ax is not axes[0]:  # Remove the y-axis label for subplots other than the first for clarity
                # ax.set_ylabel(f'{updated_metric} ({unit} $\\times 10^{exponent}$)') if exponent > 2 or exponent < -2 else ax.set_ylabel(f'{updated_metric} ({unit})')
                ax.set_title(f'{updated_metric} ({unit} $\\times 10^{exponent}$)', fontsize=11) if exponent > 2 or exponent < -2 else ax.set_title(f'{updated_metric} ({unit})', fontsize=11)

        # Remove the frame around each subplot
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes:  # Enable bottom spine for all but the last subplot
        ax.spines['bottom'].set_visible(True)
    for ax in axes:  # Enable top spine for all but the first subplot
        ax.spines['top'].set_visible(True)
    # axes[0].spines['left'].set_visible(True)
    # axes[-1].spines['right'].set_visible(True)

# Add a super title to the entire figure
fig.text(0.5, 0.95, 'Cardiac (Systole)', ha='center', fontsize=11, weight='bold')
fig.text(0.5, 0.45, 'Carotid', ha='center', fontsize=11, weight='bold')
# fig.suptitle('Cardiac', fontsize=8)
# fig.suptitle('Carotid', y=0.45, fontsize=8)
# fig.suptitle('Comparison of Algorithms Across Multiple Metrics', fontsize=11)

# Manually add a legend to the figure
handles = [plt.Line2D([0], [0], color='blue', lw=4),
           plt.Line2D([0], [0], color='red', lw=4)]
labels = ['Baseline', 'Co-Registration']
fig.legend(handles, labels, loc='upper right', title='Algorithm', bbox_to_anchor=(1, 1), ncol=2, fontsize=11)

# Adjust the layout and display the plot
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.90])  # Adjust the tight_layout to consider the legend and increase spacing
plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.4)  # Make space for the super title
# plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Results/boxplot_live_02.png', dpi=600)
plt.show()



## Plot 5: Line plots of mean error and standard deviation over different parameters and the SSD ove different parameters

# Load the CSV file into a DataFrame
df = pd.read_csv(f'{path}/{dataset}/{batch}/all_results.csv')

# Function to process and return grouped data
def process_algorithm_data(df, algorithm_name):
    # Filter the DataFrame to include only the specified algorithm
    filtered_df = df[df['Algorithm'] == algorithm_name]

    # Remove rows with NaN values in the 'Beta' column
    filtered_df = filtered_df.dropna(subset=['Beta'])

    # Ensure 'Beta' column is correctly typed
    filtered_df['Beta'] = pd.to_numeric(filtered_df['Beta'], errors='coerce')

    # Group by the 'Beta' parameter and calculate the mean and standard deviation of mean errors and SSD
    grouped = filtered_df.groupby('Beta').agg(
        mean_error=('Mean Error', 'mean'),
        std_error=('Mean Error', 'std'),
        ssd_mean=('SSD', 'mean'),
        ssd_std=('SSD', 'std')
    ).reset_index()
    
    return grouped

# Process data for both 'dipy_custom' and 'dipy' algorithms
grouped_custom = process_algorithm_data(df, 'dipy_custom')
grouped_dipy = process_algorithm_data(df, 'dipy')

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot 'dipy_custom' Mean Error
param_values_custom = grouped_custom['Beta'] * 100
mean_errors_custom = grouped_custom['mean_error'] * 1000000
std_errors_custom = grouped_custom['std_error'] * 1000000

# Plot 'dipy' Mean Error
param_values_dipy = grouped_dipy['Beta'] * 100
mean_errors_dipy = grouped_dipy['mean_error'] * 1000000
std_errors_dipy = grouped_dipy['std_error'] * 1000000

ax1.plot(param_values_custom, mean_errors_custom, label='Mean Error', color='blue')
ax1.fill_between(param_values_custom, 
                 mean_errors_custom,
                #  mean_errors_custom - std_errors_custom + std_errors_dipy, 
                 mean_errors_custom + std_errors_custom - std_errors_dipy, 
                 color='blue', alpha=0.2)

ax1.plot(param_values_dipy, mean_errors_dipy, label='Baseline Mean Error', color='blue', linestyle='dashed')
# ax1.fill_between(param_values_dipy, 
#                  mean_errors_dipy - std_errors_dipy, 
#                  mean_errors_dipy + std_errors_dipy, 
#                  color='red', alpha=0.2)

# Set labels and title for ax1
ax1.set_xlabel('β (%)', fontsize=11)
ax1.set_ylabel('Absolute Error (\u00B5m)', color='blue', fontsize=11)
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for SSD
ax2 = ax1.twinx()

# Plot 'dipy_custom' SSD
ssd_mean_custom = grouped_custom['ssd_mean'] / 1000
ssd_std_custom = grouped_custom['ssd_std'] / 1000

# Plot 'dipy' SSD
ssd_mean_dipy = grouped_dipy['ssd_mean'] / 1000
ssd_std_dipy = grouped_dipy['ssd_std'] / 1000

ax2.plot(param_values_custom, ssd_mean_custom, label='SSD', color='green')
ax2.fill_between(param_values_custom, 
                 ssd_mean_custom,
                #  ssd_mean_custom - ssd_std_custom + ssd_std_dipy, 
                 ssd_mean_custom + ssd_std_custom - ssd_std_dipy, 
                 color='green', alpha=0.2)

ax2.plot(param_values_dipy, ssd_mean_dipy, label='Baseline SSD', color='green', linestyle='dashed')
# ax2.fill_between(param_values_dipy, 
#                  ssd_mean_dipy - ssd_std_dipy, 
#                  ssd_mean_dipy + ssd_std_dipy, 
#                  color='orange', alpha=0.2)

# Set labels for ax2
ax2.set_ylabel('SSD ($\\times 10^3$)', color='green', fontsize=11)
ax2.tick_params(axis='y', labelcolor='green')

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, ["Absolute Error \u00B1 \u0394 std", 'Baseline Absolute Error'] + ["SSD \u00B1 \u0394 std", 'Baseline SSD'], loc='center left', fontsize=11)

# Display the plot
plt.suptitle('Effect of β Parameter on Absolute Error and SSD', fontsize=12)
plt.tight_layout()
# plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Results/effect_beta_live_02.png', dpi=600)
plt.show()

## Plot 6: Line plot of dice over frames for all datasets

# Load the datasets
df1 = pd.read_csv('/Users/elliottunstall/Desktop/Imperial/FYP/Results/cardiac/systole/Alpha-0_Beta-1.csv')
df2 = pd.read_csv('/Users/elliottunstall/Desktop/Imperial/FYP/Results/cardiac/diastole/Alpha-0_Beta-1.csv')
df3 = pd.read_csv('/Users/elliottunstall/Desktop/Imperial/FYP/Results/soft/10/Alpha-0_Beta-1.csv')

# Normalize the frame numbers to start from 0
df1['Normalized Frame'] = (df1['Frame'] - df1['Frame'].min()) + 1
df2['Normalized Frame'] = (df2['Frame'] - df2['Frame'].min()) + 1
df3['Normalized Frame'] = (df3['Frame'] - df3['Frame'].min()) + 1

frame_length = min(df1['Normalized Frame'].max(), df2['Normalized Frame'].max(), df3['Normalized Frame'].max())
frames = np.arange(frame_length + 1)

# Plotting
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=False)

# Subplot 1: Dice Score vs. Frames for Different Datasets
# Plot Dataset 1
ax1.plot(df1['Normalized Frame'], df1['DICE'], label='Cardiac - Diastole', color=husl[0], linewidth=2, marker='x')
# Plot Dataset 2
ax1.plot(df2['Normalized Frame'], df2['DICE'], label='Cardiac - Systole', color=husl[1], linewidth=2, marker='x')
# Plot Dataset 3
ax1.plot(df3['Normalized Frame'][0:38], df3['DICE'][0:38], label='Carotid', color=husl[2], linewidth=2, marker='x')

# Adding labels and title
ax1.set_xlabel('Relative Frame Number', fontsize=10)
ax1.set_ylabel('Dice Score', fontsize=10)
# ax1.set_title('Dice Score vs. Frames for Different Datasets')
ax1.legend(fontsize=10)

# Format the x-axis
ax1.set_xlim(left=1)
ax1.xaxis.set_major_locator(MultipleLocator(2))  # Set major ticks at intervals of 0.1
ax1.xaxis.set_minor_locator(MultipleLocator(1))  # Set minor ticks at intervals of 0.05
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Format the major tick labels
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines

## Subplot 2: HD vs. Frames for Different Datasets
# Plot Dataset 1
ax2.plot(df1['Normalized Frame'], df1['HD']*1000, label='Cardiac - Diastole', color=husl[0], linewidth=2, marker='x')
# Plot Dataset 2
ax2.plot(df2['Normalized Frame'], df2['HD']*1000, label='Cardiac - Systole', color=husl[1], linewidth=2, marker='x')
# Plot Dataset 3
ax2.plot(df3['Normalized Frame'][0:38], df3['HD'][0:38]*1000, label='Carotid', color=husl[2], linewidth=2, marker='x')

# Adding labels and title
ax2.set_xlim(left=1)
ax2.set_xlabel('Relative Frame Number', fontsize=10)
ax2.set_ylabel('Hausdorff Distance (mm)', fontsize=10)
# ax2.set_title('Hausdorff Distance vs. Frames for Different Datasets')
ax2.legend(fontsize=10)

# Format the x-axis
ax2.xaxis.set_major_locator(MultipleLocator(2))  # Set major ticks at intervals of 0.1
ax2.xaxis.set_minor_locator(MultipleLocator(1))  # Set minor ticks at intervals of 0.05
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Format the major tick labels
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines

# Display the plot
plt.suptitle('Automatic Segmentation Performance Metrics')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Results/SAS_live.png', dpi=1200)
plt.show()