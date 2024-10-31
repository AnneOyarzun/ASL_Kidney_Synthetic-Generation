import SimpleITK as sitk
import os 
import cv2

def convert_nii_folder_to_png(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all NIfTI files in the input folder
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # Loop through each NIfTI file
    for nii_file in nii_files:
        nii_file_path = os.path.join(input_folder, nii_file)
        
        # Read NIfTI file using SimpleITK
        nii_image = sitk.ReadImage(nii_file_path)

        # Convert SimpleITK image to NumPy array
        nii_data = sitk.GetArrayFromImage(nii_image)

        # # Normalize intensity values to [0, 255]
        # nii_data = ((nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255).astype(np.uint8)

        # # Convert to RGB format (assuming it's a single-channel grayscale image)
        # nii_rgb = cv2.cvtColor(nii_data, cv2.COLOR_GRAY2RGB)

        # Save as PNG
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(nii_file)[0]}.png")
        cv2.imwrite(output_file_path, nii_data)