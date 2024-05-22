import pandas as pd
import os
import time
from registration_propagation import run

frames = list(range(2, 21)) + list(range(35, 61))

# Create an empty DataFrame with columns for metrics and settings
columns = ['Frame', 'Dataset', 'Algorithm', 'Parameters', 'Time', 'DICE', 'HD', 'BCE', 'SSD', 'pSSD', 'MSE', 'pMSE', 'MI', 'pMI', 'SSIM', 'pSSIM', 'NCC', 'pNCC', 'JE', 'pJE', 'Mag Error', 'Mean Error', 'SD Error']
results_df = pd.DataFrame()

algorithms = ['dipy', 'dipy_custom']
parameters = 'Alpha-0_Beta-1'
dataset = 'Cardiac'
batch = 'Complete'

# Process each image and append results to the DataFrame
for f_indx, frame in enumerate(frames):
    start_time = time.time()
    errors, mag_error1, sd_error1, mean_error1, errors_custom, mag_error2, sd_error2, mean_error2, execution_time1, execution_time2, dice, hd, bce = run(frame)
    row = {
        'Frame': frame,
        'Dataset': dataset,
        'Algorithm': algorithms[0],
        'Parameters': parameters,
        'Time':     execution_time1,
        'DICE':     dice,
        'HD':       hd,
        'BCE':      bce,
        'SSD':      errors[0,0],
        'pSSD':     errors[0,1],
        'MSE':      errors[1,0],
        'pMSE':     errors[1,1],
        'MI':       errors[2,0],
        'pMI':      errors[2,1],
        'SSIM':     errors[3,0],
        'pSSIM':    errors[3,1],
        'NCC':      errors[4,0],
        'pNCC':     errors[4,1],
        'JE':       errors[5,0],
        'pJE':      errors[5,1],
        'Mag Error': mag_error1,
        'Mean Error': mean_error1,
        'SD Error': sd_error1  
    }
    
    # Append row to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    row = {
        'Frame': frame,
        'Dataset': dataset,
        'Algorithm': algorithms[1],
        'Parameters': parameters,
        'Time':     execution_time2,
        'DICE':     dice,
        'HD':       hd,
        'BCE':      bce,
        'SSD':      errors_custom[0,0],
        'pSSD':     errors_custom[0,1],
        'MSE':      errors_custom[1,0],
        'pMSE':     errors_custom[1,1],
        'MI':       errors_custom[2,0],
        'pMI':      errors_custom[2,1],
        'SSIM':     errors_custom[3,0],
        'pSSIM':    errors_custom[3,1],
        'NCC':      errors_custom[4,0],
        'pNCC':     errors_custom[4,1],
        'JE':       errors_custom[5,0],
        'pJE':      errors_custom[5,1],
        'Mag Error': mag_error2,
        'Mean Error': mean_error2,
        'SD Error': sd_error2,
    }

    # Append row to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    end_time = time.time()
    print(f"Frame {frame} processed in {end_time - start_time} seconds")
    print(f"{f_indx/len(frames)*100}% complete")

# Save the DataFrame to a CSV file
csv_file_path = f'/Users/elliottunstall/Desktop/Imperial/FYP/Results/{dataset}/{batch}/{parameters}.csv'
results_df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")
