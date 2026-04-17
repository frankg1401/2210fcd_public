import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os

# ===== CHANGE THIS =====
patient_folder = r'D:\TEMP\Processed\sub-00015'

def find_file(folder, suffix):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(suffix):
                return os.path.join(root, f)
    return None

# Load files
t1_path = find_file(patient_folder, 't1_norm.nii.gz')
flair_path = find_file(patient_folder, 'flair_norm.nii.gz')
mask_path = find_file(patient_folder, 'roi_registered.nii.gz')

t1 = nib.load(t1_path).get_fdata()
flair = nib.load(flair_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# ===== find best slice (max lesion) =====
slice_idx = np.argmax([np.sum(mask[:, :, i]) for i in range(mask.shape[2])])
print("Best slice:", slice_idx)

def rotate(x):
    return np.rot90(x, k=1)

# ===== create figure =====
plt.figure(figsize=(10, 4))

# T1
plt.subplot(1, 3, 1)
plt.imshow(rotate(t1[:, :, slice_idx]), cmap='gray')
plt.title("T1")
plt.axis('off')

# FLAIR
plt.subplot(1, 3, 2)
plt.imshow(rotate(flair[:, :, slice_idx]), cmap='gray')
plt.title("FLAIR")
plt.axis('off')

# FLAIR + lesion
plt.subplot(1, 3, 3)
plt.imshow(rotate(flair[:, :, slice_idx]), cmap='gray')
plt.contour(
    rotate(mask[:, :, slice_idx]),
    levels=[0.5],
    colors='white',
    linewidths=1.5
)
plt.title("FLAIR + mask")
plt.axis('off')

plt.tight_layout()

# ===== SAVE FIGURE =====
import os
save_dir = r"D:\GIT\CSC2210_Project\2210fcd\figures"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "figure1.png")

print("Saving to:", save_path)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()