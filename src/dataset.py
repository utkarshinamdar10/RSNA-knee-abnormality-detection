# Write a helper fuction to read a single DICOM slice and normalize its contrast

import pydicom
import numpy as np
import os
import cv2
from scipy import ndimage
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------
# 1. Load DICOM Slice
# ---------------------------

def load_dicom_slice(filepath):
    """Loads a single DICOM slice, applies slope/intercept scaling,
    and normalizes the pixel values between 0.0 and 1.0.
    """
    dcm = pydicom.dcmread(filepath)
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    # 1. Apply Rescale Slope and Intercept standard HU scaling if present
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    pixel_array = pixel_array * slope + intercept
    
    # 2. Min-Max normalization to [0.0, 1.0] range
    denom = (pixel_array.max() - pixel_array.min())
    if denom == 0:
        return np.zeros_like(pixel_array)
    normalized_image = (pixel_array - pixel_array.min()) / denom
    
    return normalized_image

# ---------------------------
# 2. Load and Resize Series
# ---------------------------

def load_and_resize_series(series_dir, target_slices=32, target_size=(256, 256)):
    """Loads all DICOM slices in a series, sorts them, stacks them into a 3D volume,

    and resizes/resamples the volume to the target dimensions.
    """
    # 1. Get all DICOM file paths in the directory
    dcm_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.dcm')]
    if not dcm_files:
        raise ValueError(f"No DICOM files found in directory: {series_dir}")
        
    # 2. Sort files anatomically based on Z-position in ImagePositionPatient metadata
    def get_z_position(filepath):
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            return float(dcm.ImagePositionPatient[2])
        except Exception:
            # Fallback to sorting by file name if metadata is missing
            return float(os.path.splitext(os.path.basename(filepath))[0])
            
    dcm_files.sort(key=get_z_position)
    
    # 3. Load all slices into a list and stack into 3D volume
    slices = [load_dicom_slice(f) for f in dcm_files]
    volume = np.stack(slices, axis=0)  # Shape: (num_slices, height, width)
    
    # 4. Resize each 2D slice/frame to target size (height, width)
    resized_slices = []
    for slice_img in volume:
        resized_slice = cv2.resize(slice_img, target_size, interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)
    volume = np.stack(resized_slices, axis=0)
    
    # 5. Resample slice count (depth) to exactly target_slices (32)
    current_slices = volume.shape[0]
    if current_slices != target_slices:
        indices = np.linspace(0, current_slices - 1, target_slices)
        # 1D interpolation along the slice (depth) axis
        volume = ndimage.map_coordinates(volume, [indices, None, None], order=1)
        
    return volume

# ---------------------------
# 3. Create Custom Dataset
# ---------------------------

class RSNAKneeDataset(Dataset):
    """PyTorch Dataset for loading 3D MRI volumes and target labels."""
    def __init__(self, csv_file, series_csv, data_dir, plane='Sagittal', target_slices=32, target_size=(256, 256), is_train=True):
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.target_size = target_size
        self.is_train = is_train
        
        # Load CSV mappings
        self.labels_df = pd.read_csv(csv_file)
        self.series_df = pd.read_csv(series_csv)
        
        # Filter series metadata by the desired anatomical plane
        self.plane_series = self.series_df[self.series_df['Anatomical_Plane'].str.lower() == plane.lower()]
        self.studies = self.plane_series['StudyInstanceUID'].unique()
        
        # Target abnormality columns
        self.label_cols = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        
    def __len__(self):
        return len(self.studies)
        
    def __getitem__(self, idx):
        study_uid = self.studies[idx]
        
        # Get the corresponding series for this study matching the plane
        study_series = self.plane_series[self.plane_series['StudyInstanceUID'] == study_uid]
        series_uid = study_series.iloc[0]['SeriesInstanceUID']
        
        # Match directory subfolder name (train_series vs test_series)
        subfolder = 'train_series' if self.is_train else 'test_series'
        series_dir = os.path.join(self.data_dir, subfolder, study_uid, series_uid)
        
        # Load and process the 3D volume
        try:
            volume = load_and_resize_series(series_dir, self.target_slices, self.target_size)
        except Exception as e:
            # Fallback to zero-volume if directory is missing/corrupted
            volume = np.zeros((self.target_slices, *self.target_size), dtype=np.float32)
            
        # Add channel dimension: (1, depth, H, W)
        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        
        # Load targets (fill with dummy values if in test/prediction mode)
        study_labels = self.labels_df[self.labels_df['StudyInstanceUID'] == study_uid]
        if len(study_labels) > 0 and self.is_train:
            targets = study_labels.iloc[0][self.label_cols].values.astype(np.float32)
        else:
            targets = np.zeros(len(self.label_cols), dtype=np.float32)
            
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        return volume_tensor, targets_tensor


# ---------------------------
# 4. Test Dataset
# ---------------------------

if __name__ == '__main__':
    print("=== Testing RSNAKneeDataset ===")
    
    # Setup paths (local relative paths)
    csv_file = "data/test.csv"
    series_csv = "data/test_series.csv"
    data_dir = "data"
    
    # Instantiate the dataset for the Axial plane (the downloaded sample slice is Axial)
    # is_train=False reads from the 'test_series' directory
    dataset = RSNAKneeDataset(
        csv_file=csv_file,
        series_csv=series_csv,
        data_dir=data_dir,
        plane='Axial',
        target_slices=32,
        target_size=(256, 256),
        is_train=False
    )
    
    print(f"Dataset length (number of studies with Axial plane): {len(dataset)}")
    
    # Retrieve the first sample
    volume, targets = dataset[0]
    
    print("\n--- Output Verification ---")
    print(f"Volume Tensor Shape: {volume.shape} (Expected: torch.Size([1, 32, 256, 256]))")
    print(f"Volume Tensor Dtype: {volume.dtype} (Expected: torch.float32)")
    print(f"Targets Tensor Shape: {targets.shape} (Expected: torch.Size([12]))")
    print(f"Targets: {targets}")
    print("\n=== Test Complete ===")
