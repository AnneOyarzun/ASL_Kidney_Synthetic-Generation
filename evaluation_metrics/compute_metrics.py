import os
import pandas as pd
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import cv2
import numpy as np

def compute_metrics(real_folder, fake_folder):
    metric_values = []

    real_images = os.listdir(real_folder)
    fake_images = os.listdir(fake_folder)

    for real_image_name in real_images:
        if real_image_name in fake_images:
            real_image_path = os.path.join(real_folder, real_image_name)
            fake_image_path = os.path.join(fake_folder, real_image_name)
            real_image = sitk.ReadImage(real_image_path)
            real_image_array = sitk.GetArrayFromImage(real_image)
            real_image_array = (real_image_array - real_image_array.min()) / (real_image_array.max() - real_image_array.min())

            fake_image = sitk.ReadImage(fake_image_path)
            fake_image_array = sitk.GetArrayFromImage(fake_image)
            fake_image_array = (fake_image_array - fake_image_array.min()) / (fake_image_array.max() - fake_image_array.min())

            # Structural Similarity Index (SSIM)
            similarity = ssim(real_image_array, fake_image_array, channel_axis=2, data_range=1.0)
            
            # Feature Similarity Index Measure (FSIM)
            fsim_value = fsim(real_image_array, fake_image_array)
            
            # Mean Absolute Error (MAE)
            mae_value = np.mean(np.abs(real_image_array - fake_image_array))
            
            # Error Per Pixel (EPR)
            epr_value = calculate_epr(real_image_array, fake_image_array)
            
            # # Mean Squared Error (MSE)
            # mse_value = mean_squared_error(real_image_array, fake_image_array)
            
            # # Peak Signal-to-Noise Ratio (PSNR)
            # psnr_value = peak_signal_noise_ratio(real_image_array, fake_image_array, data_range=1.0)
            
            metric_values.append([real_image_name, similarity, fsim_value, mae_value, epr_value])

    return metric_values

# Define FSIM function
def fsim(img1, img2, K=0.02, sigma=3):
    g1 = cv2.GaussianBlur(img1, (11, 11), sigma)
    g2 = cv2.GaussianBlur(img2, (11, 11), sigma)
    M1 = 2 * g1 * g2 + K
    M2 = g1**2 + g2**2 + K
    return np.mean(M1 / M2)

# Function to calculate Edge Preservation Ratio (EPR)
def calculate_epr(real_image_array, fake_image_array):
    # Calculate absolute difference between real and fake images
    abs_diff = np.abs(real_image_array - fake_image_array)
    # Normalize the absolute difference to [0, 1]
    normalized_diff = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min())
    # Calculate the mean value as EPR
    return np.mean(normalized_diff)


