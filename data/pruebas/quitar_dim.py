import SimpleITK as sitk
import os

path = 'D:/RM_RENAL/CycleGAN/dataset/testB/'
files = os.listdir(path)

for i in range (0, len(files)): 
    img = sitk.ReadImage(path + files[i])
    img_new = img[:,:,0]
    sitk.WriteImage(img_new, path + files[i])