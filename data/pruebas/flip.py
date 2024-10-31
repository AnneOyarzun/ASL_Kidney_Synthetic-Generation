import SimpleITK as sitk
import os

def flip_images_in_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = sitk.ReadImage(image_path)

        # Flip the image vertically
        flipped_image = sitk.Flip(image, [False, True, False])
        flipped_image.SetOrigin(image.GetOrigin())

        # Save the flipped image to the output folder
        output_path = os.path.join(output_folder, filename)
        sitk.WriteImage(flipped_image, output_path)

# Specify the input and output folders
input_folder = 'D:/RM_RENAL/CycleGAN/dataset/no_flip/'
output_folder = 'D:/RM_RENAL/CycleGAN/dataset/flip/'

# Call the function to flip images in the input folder and save them to the output folder
flip_images_in_folder(input_folder, output_folder)

