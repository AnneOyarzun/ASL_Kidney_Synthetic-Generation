import os
import numpy as np
import SimpleITK as sitk

def convert_png_to_nifti(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # Loop through each PNG file
    for png_file in png_files:
        # Read the PNG file
        png_path = os.path.join(input_folder, png_file)
        png_image = sitk.ReadImage(png_path)

        # Convert to NIfTI format
        nifti_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(png_image))
        nifti_image.CopyInformation(png_image)

        # Save as NIfTI file
        nifti_filename = os.path.splitext(png_file)[0] + '.nii'
        nifti_path = os.path.join(output_folder, nifti_filename)
        sitk.WriteImage(nifti_image, nifti_path)

# Example usage
input_folder = 'datasets_rm/testB/'
output_folder = 'datasets_rm/testB_nii/'
convert_png_to_nifti(input_folder, output_folder)
