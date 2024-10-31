import os
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def load_nifti_image(file_path):
    return nib.load(file_path).get_fdata()

def normalize_image(image):
    # Normalize image to range 0-255
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = ((image - image_min) / (image_max - image_min)) * 255
    return normalized_image.astype(np.uint8)

def resize_image(image, new_shape=(256, 256), order=3):
    """
    Resize the image to new_shape using specified order interpolation.
    Order 0: Nearest-neighbor (useful for binary masks)
    Order 3: Cubic interpolation (default for general images)
    """
    zoom_factor = np.array(new_shape) / np.array(image.shape)
    resized_image = zoom(image, zoom_factor, order=order)
    return resized_image.astype(np.float32)

def calculate_histogram(image):
    # Ignore NaN values in the histogram calculation
    hist = np.histogram(image[~np.isnan(image)].ravel(), bins=256, range=(0, 256))[0]
    # Normalize the histogram
    hist = hist / np.sum(hist)
    return hist

def calculate_histcc(hist1, hist2):
    return pearsonr(hist1, hist2)[0]

def process_images(real_dir, fake_dir):
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.nii')]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.nii')]

    histcc_values = []

    for real_file, fake_file in zip(real_files, fake_files):
        try:
            real_image = load_nifti_image(real_file)
            fake_image = load_nifti_image(fake_file)[0, :, :]

            # Normalize and resize images
            real_image = normalize_image(real_image)
            real_image = resize_image(real_image, new_shape=(256, 256))
            fake_image = normalize_image(fake_image)
            fake_image = resize_image(fake_image, new_shape=(256, 256))

            real_kidney_region = real_image
            fake_kidney_region = fake_image

            # Ensure there is variation in the kidney regions
            if np.all(np.isnan(real_kidney_region)) or np.all(np.isnan(fake_kidney_region)):
                print(f"Skipping {real_file} and {fake_file} due to NaN regions.")
                continue
            
            if np.nanstd(real_kidney_region) == 0 or np.nanstd(fake_kidney_region) == 0:
                print(f"Skipping {real_file} and {fake_file} due to constant regions.")
                continue

            # Calculate histograms for the kidney regions
            real_histogram = calculate_histogram(real_kidney_region)
            fake_histogram = calculate_histogram(fake_kidney_region)

            # Calculate HistCC between real and fake histograms
            hist_cc = calculate_histcc(real_histogram, fake_histogram)
            histcc_values.append(hist_cc)

        except Exception as e:
            print(f"Error processing {real_file} and {fake_file}: {e}")
            continue

    return histcc_values

if __name__ == "__main__":
    
    histcc_values = process_images(real_dir, fake_dir)
    if histcc_values:
        average_histcc = np.mean(histcc_values)
        print(f"Fake directory is {fake_path} → Average HistCC: {average_histcc}")
    else:
        print("No valid HistCC values were computed.")

