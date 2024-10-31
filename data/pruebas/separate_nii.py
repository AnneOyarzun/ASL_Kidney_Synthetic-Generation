import SimpleITK as sitk
import os

def save_slices_as_nii(input_file, output_dir):
    # Load the NIfTI file
    nifti_img = sitk.ReadImage(input_file)
    data = sitk.GetArrayFromImage(nifti_img)

    # Extracting the name of the input file (without extension) to name individual slices
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Create a directory to save the slices if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each slice as a separate NIfTI file
    for i in range(0, data.shape[0]):
        slice_data = data[i, :, :]
        slice_img = sitk.GetImageFromArray(slice_data)
        # slice_img.CopyInformation(nifti_img)
        slice_filename = os.path.join(output_dir, f"{base_name}_slice_{i}.nii")
        sitk.WriteImage(slice_img, slice_filename)

    print(f"Saved {data.shape[-1]} slices as NIfTI files in {output_dir}")

# Usage example:
input_file = 'D:/RM_RENAL/CycleGAN/dataset/trainA/V03_01_A_EP2D_SE_PCASL_CENTRIC_KIDNEY_SIN_BS_0015_1_51.nii' # Provide the path to your NIfTI file
output_dir = 'D:/RM_RENAL/CycleGAN/dataset/trainA/'
save_slices_as_nii(input_file, output_dir)
