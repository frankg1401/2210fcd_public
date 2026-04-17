# High Level Description: This is a simple script to read and process the full mri data in a usuable format to feed into CNN
# This script will perform the following tasks:

# 1) function to read in each sub-# folder's T1 and FLAIR datasets as tensors
# 2) function to visualize slices of x patients FLAIR or T1 sequence and with labeled masked lesion

# Credit  Daniel Rafique

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 

# Example path 
patient_folder = '/workspace/MRI_data/sub-00001/anat'

def find_file_by_suffix(folder, suffix): # since there is some inconsistnecy with middle of the path files go by suffix (T1w.nii, FLAIR.nii.gz ...)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                return os.path.join(root, file)
    return None

def load_patient(patient_folder, getsPaths = False):

    '''
    This function is a patient loader that takes a patient folder path
    and then returns t1, flair, and flair_roi mask loaded data
    Optinally, you can return the paths of the t1, flair, and roi instead of the nib loaded versions 
    '''
    t1_path = find_file_by_suffix(patient_folder, 'T1w.nii.gz')
    flair_path = find_file_by_suffix(patient_folder, 'FLAIR.nii.gz')
    
    # try both common naming patterns for the mask
    mask_path = (find_file_by_suffix(patient_folder, 'FLAIR_roi.nii.gz') or
                 find_file_by_suffix(patient_folder, 'roi_flair.nii.gz'))
    
    if not (t1_path and flair_path):
        raise FileNotFoundError(f"Missing one or more required files in: {patient_folder}")
    
    if getsPaths == True:
        return t1_path, flair_path, mask_path #mask path might be none
    
    t1 = nib.load(t1_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

   
    return t1, flair, mask

def visualize_best_slice(mask, flair):
    '''
    This function will return a visualization of the slice with the most amount of positive mask pixels

    '''
    # Find the slice with the most non-zero mask pixels
    sums = [np.sum(mask[:, :, i]) for i in range(mask.shape[2])]
    best_slice = np.argmax(sums)
    print("Best slice:", best_slice)

    # Visualize it
    plt.subplot(1,3,3)
    plt.imshow(rotate_90_ccw(flair[:, :, best_slice]), cmap='gray')
    plt.imshow(rotate_90_ccw(mask[:, :, best_slice]), cmap='Reds', alpha=0.4)
    plt.title(f"FLAIR + ROI (Slice {best_slice})")
    plt.axis('off')
    plt.show()

def rotate_90_ccw(slice_2d): # function to rotate a 2d slice 90 degrees counter clockwise (for visualization purposes)
    return np.rot90(slice_2d, k=1)

#### driver code to test loaded files ####

# Load images
#t1 = nib.load(os.path.join(patient_folder, 'sub-00001_acq-iso08_T1w.nii.gz')).get_fdata()
#flair = nib.load(os.path.join(patient_folder, 'sub-00001_acq-T2sel_FLAIR.nii.gz')).get_fdata()
#mask = nib.load(os.path.join(patient_folder, 'sub-00001_acq-T2sel_FLAIR_roi.nii.gz')).get_fdata()

#testing
def test_visualization():

    t1, flair, mask = load_patient(patient_folder)
    # Check shapes and mask values
    print("T1 shape:", t1.shape)
    print("FLAIR shape:", flair.shape)
    print("Mask shape:", mask.shape)
    print("Mask unique values:", np.unique(mask))  # Should be [0. 1.] or [0 1]

    # Visualize middle slice
    mid = t1.shape[2] // 2
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rotate_90_ccw(t1[:, :, mid]), cmap='gray')
    plt.title("T1")

    plt.subplot(1, 3, 2)
    plt.imshow(rotate_90_ccw(flair[:, :, mid]), cmap='gray')
    plt.title("FLAIR")

    # show best slice where the lesion is highlighted on roi
    visualize_best_slice(mask, flair)

    plt.tight_layout()
    plt.show()

    print("Unique mask values:", np.unique(mask))

#test_visualization()


#load_patient(patient_folder)