import mat73
import numpy as np

print("Loading data...")
images1 = mat73.loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/Beamformed_Acquisition_1.mat')
print("50%")
images2 = mat73.loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/Beamformed_Acquisition_2.mat')
print("100%")
print("Data loaded")
print(f"shape: {np.shape(images1)}")