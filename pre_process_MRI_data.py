# This alternate script should pre-process FLAIR, T1, and ROI files to usable states for training
# Credit  Daniel Rafique 

import ants
import numpy as np
import nibabel as nib
import os
import reading_MRI_data as read_MRI
import matplotlib.pyplot as plt

def preprocess_with_ants(t1_path, flair_path, roi_path, crop_shape=(160,192,160), save_dir=None, file_suffix = None):
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    roi_reg = None
    # 1. Load images with ANTs
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)

    if roi_path!= None:
        roi = ants.image_read(roi_path)

    #print(f'raw t1 loaded shape: {t1.shape}')
    #print(f'raw flair loaded shape: {flair.shape}')
    #print(f'raw roi loaded shape: {roi.shape}')

    # Step 2: Resample all to 1x1x1 mm (before registration)
    t1_res = ants.resample_image(t1, resample_params=(1,1,1), use_voxels=False)
    flair_res = ants.resample_image_to_target(flair, t1_res)
    if roi_path!= None:
        roi_res = ants.resample_image_to_target(roi, t1_res, interp_type=0)  # Nearest-neighbor

    #print(f'resampled shape: {t1_res.shape}')
    #print(f'resampled flair shape: {flair_res.shape}')
    #print(f'resampled roi shape: {roi_res.shape}')

    # function to Crop to center region (same for all)
    def center_crop(ant_img, target_shape):
        img_np = ant_img.numpy()
        d, h, w = img_np.shape
        td, th, tw = target_shape

        start_d = max((d - td) // 2, 0)
        start_h = max((h - th) // 2, 0)
        start_w = max((w - tw) // 2, 0)

        cropped_np = img_np[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]
        cropped = ants.from_numpy(cropped_np, spacing=ant_img.spacing)
        cropped.set_origin(ant_img.origin)
        cropped.set_direction(ant_img.direction)
        return cropped

    t1_crop = center_crop(t1_res, crop_shape)
    flair_crop = center_crop(flair_res, crop_shape)

    if roi_path!= None:
        roi_crop = center_crop(roi_res, crop_shape)

    #print(f'cropped shape: {t1_crop.shape}')
    #print(f'cropped flair shape: {flair_crop.shape}')
    #print(f'cropped roi shape: {roi_crop.shape}')

    # 3. Register FLAIR → T1
    reg = ants.registration(fixed=t1_crop, moving=flair_crop, type_of_transform='Rigid')
    flair_reg = reg['warpedmovout']

    # 4. Apply transform to ROI
    if roi_path!= None:
        roi_reg = ants.apply_transforms(fixed=t1_crop, moving=roi_crop,
                                        transformlist=reg['fwdtransforms'],
                                        interpolator='nearestNeighbor')

    # 5. Normalize (z-score within nonzero mask)
    def normalize_zscore(img):
        arr = img.numpy()
        mask = arr > 0
        mean = arr[mask].mean()
        std = arr[mask].std()
        arr_norm = np.zeros_like(arr)
        arr_norm[mask] = (arr[mask] - mean) / (std + 1e-5)
        return ants.from_numpy(arr_norm, spacing=img.spacing, origin=img.origin, direction=img.direction)

    t1_norm = normalize_zscore(t1_crop)
    flair_norm = normalize_zscore(flair_reg)
    #print(f'final normalized shape: {t1_norm.shape}')
    #print(f'final flair normalized shape: {flair_norm.shape}')
    #print(f'final roi normalized shape: {roi_reg.shape}')

    # 6. Save if requested
    if save_dir:
        ants.image_write(t1_norm, os.path.join(save_dir, file_suffix+'_processed_t1_norm.nii.gz'))
        ants.image_write(flair_norm, os.path.join(save_dir, file_suffix+'_processed_flair_norm.nii.gz'))

        if roi_path!= None:
            ants.image_write(roi_reg, os.path.join(save_dir, file_suffix+'_processed_roi_registered.nii.gz'))

    
    return t1_norm, flair_norm, roi_reg

def visualize_slice_with_roi(t1_img, flair_img, roi_img, slice_index=None, axis='z', title_prefix=''):
    """
    Visualize a single 2D slice from T1, FLAIR, and FLAIR+ROI overlay.

    Parameters:
    - t1_img, flair_img, roi_img: ANTs images (already normalized, registered, cropped)
    - slice_index: optional slice index; if None, will use center slice
    - axis: 'x', 'y', or 'z' — which plane to slice
    - title_prefix: optional title prefix for each subplot

    Returns:
    - Matplotlib figure
    """
    # Convert to NumPy
    t1_np = t1_img.numpy()
    flair_np = flair_img.numpy()
    roi_np = roi_img.numpy()

    # Determine slice index
    if slice_index is None:
        if axis == 'z':
            slice_index = t1_np.shape[0] // 2
        elif axis == 'y':
            slice_index = t1_np.shape[1] // 2
        elif axis == 'x':
            slice_index = t1_np.shape[2] // 2
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Extract slices
    if axis == 'z':
        t1_slice = t1_np[slice_index, :, :]
        flair_slice = flair_np[slice_index, :, :]
        roi_slice = roi_np[slice_index, :, :]
    elif axis == 'y':
        t1_slice = t1_np[:, slice_index, :]
        flair_slice = flair_np[:, slice_index, :]
        roi_slice = roi_np[:, slice_index, :]
    elif axis == 'x':
        t1_slice = t1_np[:, :, slice_index]
        flair_slice = flair_np[:, :, slice_index]
        roi_slice = roi_np[:, :, slice_index]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap_gray = 'gray'
    cmap_roi = 'Reds'

    axes[0].imshow(t1_slice.T, cmap=cmap_gray, origin='lower')
    axes[0].set_title(f"{title_prefix}T1 - Slice {slice_index} ({axis}-axis)")
    axes[0].axis('off')

    axes[1].imshow(flair_slice.T, cmap=cmap_gray, origin='lower')
    axes[1].set_title(f"{title_prefix}FLAIR - Slice {slice_index} ({axis}-axis)")
    axes[1].axis('off')

    axes[2].imshow(flair_slice.T, cmap=cmap_gray, origin='lower')
    axes[2].imshow(roi_slice.T, cmap=cmap_roi, alpha=0.4, origin='lower')
    axes[2].set_title(f"{title_prefix}FLAIR + ROI Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

### driver code ###

# Loop over each patient folder sub-xxxx and save new files in processed directory 
# Sample code to test patient 00001 
#t1_path, flair_path, roi_path = read_MRI.load_patient('/workspace/MRI_data/sub-00002/anat', getsPaths=True)
#t1_norm, flair_norm, roi_reg = preprocess_with_ants(t1_path, flair_path, roi_path, crop_shape=(166, 256,256), save_dir="Processed/sub-00002", file_suffix="sub-00002")

if __name__ == "__main__":
    
    for i in range(1,171):
        full_filepath = f"/workspace/MRI_data/sub-{i:05d}/anat"
        patient_num = f"{i:05d}"
        print(full_filepath)
        t1_path, flair_path, roi_path = read_MRI.load_patient(full_filepath, getsPaths=True)
        t1_norm, flair_norm, roi_reg = preprocess_with_ants(t1_path, flair_path, roi_path, crop_shape=(166, 256,256), save_dir="Processed/sub-"+patient_num, file_suffix="sub-"+patient_num)



###----------------- Visualization Testing ------------------------- ###
#t1_raw = ants.image_read(t1_path)
#flair_raw = ants.image_read(flair_path)
#roi_raw = ants.image_read(roi_path)

"""

visualize_slice_with_roi(
    t1_img=t1_raw,
    flair_img=flair_raw,
    roi_img=roi_raw,
    slice_index=116,    # Or None for center
    axis='x',          # 'x' or 'y' also allowed
    title_prefix='Patient 1 - '
)

visualize_slice_with_roi(
    t1_img=t1_norm,
    flair_img=flair_norm,
    roi_img=roi_reg,
    slice_index=116,    # Or None for center
    axis='x',          # 'x' or 'y' also allowed
    title_prefix='Patient 1 - '
)

"""