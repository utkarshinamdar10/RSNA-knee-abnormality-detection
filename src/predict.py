import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import RSNAKneeModel, LABEL_COLS
from dataset import RSNAKneeDataset

# ── Config ────────────────────────────────────────────────────────────────────
# Update DATA_DIR and CHECKPOINT_PATH when running on Kaggle Notebooks

CONFIG = {
    'DATA_DIR'       : 'data',
    'TEST_CSV'       : 'data/test.csv',
    'SERIES_CSV'     : 'data/test_series.csv',
    'CHECKPOINT_PATH': 'model_best.pt',
    'SUBMISSION_PATH': 'submission.csv',
    'PLANE'          : 'Sagittal',
    'TARGET_SLICES'  : 24,
    'IMG_SIZE'       : 224,
    'BACKBONE'       : 'efficientnet_b0',
    'BATCH_SIZE'     : 4,
    'DEVICE'         : 'cuda' if torch.cuda.is_available() else 'cpu',
}
# ─────────────────────────────────────────────────────────────────────────────


def predict():
    cfg = CONFIG
    device = torch.device(cfg['DEVICE'])
    print(f"Using device: {device}")

    if not os.path.exists(cfg['CHECKPOINT_PATH']):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg['CHECKPOINT_PATH']}\n"
            "Run src/train.py first to generate the model checkpoint."
        )

    model = RSNAKneeModel(backbone=cfg['BACKBONE'], pretrained=False)
    model.load_state_dict(torch.load(cfg['CHECKPOINT_PATH'], map_location=device))
    model.to(device).eval()
    print(f"Loaded checkpoint: {cfg['CHECKPOINT_PATH']}")

    test_ds = RSNAKneeDataset(
        csv_file      = cfg['TEST_CSV'],
        series_csv    = cfg['SERIES_CSV'],
        data_dir      = cfg['DATA_DIR'],
        plane         = cfg['PLANE'],
        target_slices = cfg['TARGET_SLICES'],
        target_size   = (cfg['IMG_SIZE'], cfg['IMG_SIZE']),
        is_train      = False,
    )

    test_loader = DataLoader(test_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=2)
    print(f"Test studies: {len(test_ds)}")

    all_preds = []
    all_uids  = []

    with torch.no_grad():
        for volumes, _ in tqdm(test_loader, desc='inference'):
            volumes = volumes.to(device)
            logits  = model(volumes)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    all_preds = np.concatenate(all_preds, axis=0)

    # Recover study IDs in the same order the DataLoader iterated
    test_df = pd.read_csv(cfg['TEST_CSV'])
    series_df = pd.read_csv(cfg['SERIES_CSV'])
    plane_series = series_df[series_df['Anatomical_Plane'].str.lower() == cfg['PLANE'].lower()]
    ordered_uids = plane_series['StudyInstanceUID'].unique()

    submission = pd.DataFrame(all_preds, columns=LABEL_COLS)
    submission.insert(0, 'StudyInstanceUID', ordered_uids[:len(submission)])

    # Ensure all test studies appear even if some had no matching series
    sample_sub = pd.read_csv(cfg['DATA_DIR'] + '/sample_submission.csv')
    submission = sample_sub[['StudyInstanceUID']].merge(submission, on='StudyInstanceUID', how='left')
    submission[LABEL_COLS] = submission[LABEL_COLS].fillna(0.5)  # fallback for missing studies

    submission.to_csv(cfg['SUBMISSION_PATH'], index=False)
    print(f"\nSubmission saved to: {cfg['SUBMISSION_PATH']}")
    print(submission.head())


if __name__ == '__main__':
    predict()
