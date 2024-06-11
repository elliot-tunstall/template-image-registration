import pandas as pd
import os
import time
from run_full_segmentation import run

# frames = list(range(2, 21)) + list(range(35, 61))
# frames = range(2, 24)

all_results_df = pd.DataFrame()

for i in range(0,1):
    alpha = 0
    beta = 1
    print(f'Alpha: {alpha}, Beta: {beta}')

    # Create an empty DataFrame with columns for metrics and settings
    columns = ['Frame', 'Dataset', 'Algorithm', 'Parameters', 'Time', 'DICE', 'HD', 'BCE', 'SSD', 'pSSD', 'MSE', 'pMSE', 'MI', 'pMI', 'SSIM', 'pSSIM', 'NCC', 'pNCC', 'JE', 'pJE', 'Mag Error', 'Mean Error', 'SD Error']
    results_df = pd.DataFrame()

    algorithms = ['dipy', 'dipy_custom']
    parameters = f'Alpha-{alpha}_Beta-{beta}'
    dataset = 'cardiac'
    batch = 'systole'
    fixed_frame = 1
    dataset_number = 10

    if dataset == 'soft':
        frames = range(1, 24)
        fixed_frame = 0

    elif batch == 'complete':
        frames = list(range(2, 21)) + list(range(35, 61))

    elif batch == 'diastole':
        frames = range(2, 21)

    elif batch == 'systole':
        frames = range(36, 55)
        fixed_frame = 35

    # Process each image and append results to the DataFrame
    for f_indx, frame in enumerate(frames):
        start_time = time.time()
        errors, mag_error1, sd_error1, mean_error1, errors_custom, mag_error2, sd_error2, mean_error2, execution_time1, execution_time2, dice, hd, bce = run(frame, dataset, fixed_frame=fixed_frame, dataset_number=dataset_number, alpha=alpha, beta=beta)
        row = {
            'Frame': frame,
            'Dataset': dataset,
            'Algorithm': algorithms[0],
            'Parameters': parameters,
            'Alpha': alpha,
            'Beta': beta,
            'Execution Time':     execution_time1,
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
            'Alpha': alpha,
            'Beta': beta,
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
        print(f"{(f_indx+1)/len(frames)*100}% complete")

    if dataset == 'cardiac':
        path = f'/Users/elliottunstall/Desktop/Imperial/FYP/Results/{dataset}/{batch}/true_segmentation'

    elif dataset == 'soft':
        path = f'/Users/elliottunstall/Desktop/Imperial/FYP/Results/{dataset}/{dataset_number}/true_segmentation'

    # Check if the directory exists
    if not os.path.exists(path):
        # If the directory doesn't exist, create it
        os.makedirs(path)

    # Save the DataFrame to a CSV file
    csv_file_path = f'{path}/{parameters}.csv'
    results_df.to_csv(csv_file_path, index=False)

    print(f"Results saved to {csv_file_path}")

    # Concatenate the dataframes
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

# Save the concatenated DataFrame to a CSV file
# Save the DataFrame to a CSV file
csv_file_path = f'{path}/all_results.csv'
all_results_df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")

