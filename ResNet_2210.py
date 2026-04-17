# Credit  Daniel Rafique for baseline early fusion 2 channel
# File has been extensively expanded for new work for CS2210

import nibabel as nib
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import resnet18


def load_nifti_as_tensor(path, normalize=True):
    """
    Inputs: path for image, normalize=True default
    Outputs: torch tensor of shape [1, D, H, W]
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if normalize:
        data = (data - np.mean(data)) / (np.std(data) + 1e-5)
    return torch.tensor(data).unsqueeze(0)


def load_dataset(modalities, nifti_dir, tsv_path):
    """
    Inputs:
        modalities: 't1', 'flair', or 't1_flair'
        nifti_dir: root processed NIfTI directory
        tsv_path: participants.tsv path

    Outputs:
        dataset list of tuples:
        (image_tensor, tabular_tensor, label_tensor, patient_id)
    """
    if modalities not in ["t1", "flair", "t1_flair"]:
        raise ValueError("modalities must be 't1', 'flair', or 't1_flair'")

    df = pd.read_csv(tsv_path, sep='\t')
    dataset = []

    for _, row in df.iterrows():
        patient_id = row["participant_id"]

        group_str = row["group"]
        group = 1 if group_str == "fcd" else 0

        sex_str = row.get("sex", "U")
        sex = 1.0 if sex_str == "M" else 0.0

        age_scan = row.get("age_scan", 0)
        if pd.isna(age_scan):
            age_scan = 0.0

        t1_path = os.path.join(nifti_dir, patient_id, f"{patient_id}_processed_t1_norm.nii.gz")
        flair_path = os.path.join(nifti_dir, patient_id, f"{patient_id}_processed_flair_norm.nii.gz")

        if modalities == "t1":
            if not os.path.exists(t1_path):
                continue
            image_tensor = load_nifti_as_tensor(t1_path)

        elif modalities == "flair":
            if not os.path.exists(flair_path):
                continue
            image_tensor = load_nifti_as_tensor(flair_path)

        else:  # t1_flair
            if not (os.path.exists(t1_path) and os.path.exists(flair_path)):
                continue
            t1_tensor = load_nifti_as_tensor(t1_path)
            flair_tensor = load_nifti_as_tensor(flair_path)
            image_tensor = torch.cat([t1_tensor, flair_tensor], dim=0)  # [2, D, H, W]

        tabular_tensor = torch.tensor([sex, age_scan], dtype=torch.float32)
        label_tensor = torch.tensor(group, dtype=torch.long)

        dataset.append((image_tensor, tabular_tensor, label_tensor, patient_id))

    return dataset


class MONAIResNet3DWithTabular(nn.Module):
    def __init__(self, in_channels=2, tabular_dim=2, use_tabular=True):
        super().__init__()
        self.use_tabular = use_tabular

        backbone = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=True,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        old_conv = backbone.conv1
        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        with torch.no_grad():
            for c in range(in_channels):
                new_conv.weight[:, c] = old_conv.weight[:, 0]
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.conv1 = new_conv
        self.backbone = backbone

        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.pool = nn.AdaptiveAvgPool3d(1)

        final_input_dim = 512 + (tabular_dim if use_tabular else 0)
        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular=None):
        x = self.features(image)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.use_tabular and tabular is not None:
            x = torch.cat([x, tabular], dim=1)

        return self.classifier(x).squeeze(1)


class LateFusionResNet3DWithTabular(nn.Module):
    """
    Late fusion for exactly two modalities:
    channel 0 = T1
    channel 1 = FLAIR
    """
    def __init__(self, tabular_dim=2, use_tabular=True):
        super().__init__()
        self.use_tabular = use_tabular

        self.t1_encoder = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=True,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        self.flair_encoder = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=True,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        self.t1_features = nn.Sequential(*list(self.t1_encoder.children())[:-1])
        self.flair_features = nn.Sequential(*list(self.flair_encoder.children())[:-1])

        self.pool = nn.AdaptiveAvgPool3d(1)

        final_input_dim = 512 + 512 + (tabular_dim if use_tabular else 0)

        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular=None):
        if image.shape[1] != 2:
            raise ValueError(f"Late fusion expects 2 channels, got {image.shape[1]}")

        t1 = image[:, 0:1, :, :, :]
        flair = image[:, 1:2, :, :, :]

        t1_feat = self.t1_features(t1)
        t1_feat = self.pool(t1_feat)
        t1_feat = t1_feat.view(t1_feat.size(0), -1)

        flair_feat = self.flair_features(flair)
        flair_feat = self.pool(flair_feat)
        flair_feat = flair_feat.view(flair_feat.size(0), -1)

        x = torch.cat([t1_feat, flair_feat], dim=1)

        if self.use_tabular and tabular is not None:
            x = torch.cat([x, tabular], dim=1)

        return self.classifier(x).squeeze(1)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import resnet18


class LateFusionResNet3DWithTabular_MemoryOptimized(nn.Module):
    """
    Memory-optimized late fusion for exactly two modalities:
    channel 0 = T1
    channel 1 = FLAIR

    Optimizations:
    1. Activation checkpointing on each encoder branch
    2. Freeze early layers to reduce training memory
    3. Slightly smaller classifier head
    """

    def __init__(
        self,
        tabular_dim=2,
        use_tabular=True,
        use_checkpoint=True,
        freeze_until="layer2",   # options: None, "layer1", "layer2", "layer3"
    ):
        super().__init__()
        self.use_tabular = use_tabular
        self.use_checkpoint = use_checkpoint
        self.freeze_until = freeze_until

        self.t1_encoder = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=True,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        self.flair_encoder = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=True,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        self.t1_features = nn.Sequential(*list(self.t1_encoder.children())[:-1])
        self.flair_features = nn.Sequential(*list(self.flair_encoder.children())[:-1])

        self.pool = nn.AdaptiveAvgPool3d(1)

        final_input_dim = 512 + 512 + (tabular_dim if use_tabular else 0)

        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.freeze_early_layers(self.freeze_until)

    def freeze_early_layers(self, freeze_until="layer2"):
        if freeze_until is None:
            return

        valid = {None, "layer1", "layer2", "layer3"}
        if freeze_until not in valid:
            raise ValueError(f"freeze_until must be one of {valid}, got {freeze_until}")

        freeze_prefixes = ["conv1", "bn1", "layer1"]

        if freeze_until in {"layer2", "layer3"}:
            freeze_prefixes.append("layer2")

        if freeze_until == "layer3":
            freeze_prefixes.append("layer3")

        for encoder in [self.t1_encoder, self.flair_encoder]:
            for name, param in encoder.named_parameters():
                if any(name.startswith(prefix) for prefix in freeze_prefixes):
                    param.requires_grad = False

    def _forward_branch(self, branch, x):
        if self.use_checkpoint and self.training:
            x = checkpoint(branch, x, use_reentrant=False)
        else:
            x = branch(x)

        x = self.pool(x)
        x = x.flatten(1)
        return x

    def forward(self, image, tabular=None):
        if image.shape[1] != 2:
            raise ValueError(f"Late fusion expects 2 channels, got {image.shape[1]}")

        t1 = image[:, 0:1, :, :, :]
        flair = image[:, 1:2, :, :, :]

        t1_feat = self._forward_branch(self.t1_features, t1)
        flair_feat = self._forward_branch(self.flair_features, flair)

        x = torch.cat([t1_feat, flair_feat], dim=1)

        if self.use_tabular and tabular is not None:
            x = torch.cat([x, tabular], dim=1)

        return self.classifier(x).squeeze(1)