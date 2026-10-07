import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import RSNAKneeModel, LABEL_COLS
from dataset import RSNAKneeDataset

# ── Config ────────────────────────────────────────────────────────────────────
# When running on Kaggle Notebooks, update DATA_DIR to:
#   /kaggle/input/rsna-knee-abnormality-detection
# and PSEUDO_CSV to:
#   /kaggle/working/train_pseudo_labeled.csv  (after running pseudo_label_generator.py)

CONFIG = {
    'DATA_DIR'     : 'data',
    'PSEUDO_CSV'   : 'data/train_pseudo_labeled.csv',
    'SERIES_CSV'   : 'data/train_series.csv',
    'PLANE'        : 'Sagittal',
    'TARGET_SLICES': 24,
    'IMG_SIZE'     : 224,
    'BACKBONE'     : 'efficientnet_b0',
    'EPOCHS'       : 10,
    'BATCH_SIZE'   : 8,
    'LR'           : 1e-4,
    'VAL_FRAC'    : 0.2,
    'SEED'         : 42,
    'SAVE_PATH'    : 'model_best.pt',
    'DEVICE'       : 'cuda' if torch.cuda.is_available() else 'cpu',
}
# ─────────────────────────────────────────────────────────────────────────────


def get_dataloaders(cfg):
    df = pd.read_csv(cfg['PSEUDO_CSV'])

    # Drop rows where all labels are still NaN (should be none after pseudo-labeling)
    df = df.dropna(subset=LABEL_COLS, how='all').reset_index(drop=True)
    df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(float)

    study_ids = df['StudyInstanceUID'].unique()
    train_ids, val_ids = train_test_split(
        study_ids, test_size=cfg['VAL_FRAC'], random_state=cfg['SEED']
    )

    train_df = df[df['StudyInstanceUID'].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df['StudyInstanceUID'].isin(val_ids)].reset_index(drop=True)

    # Save temp CSVs so RSNAKneeDataset can read them
    os.makedirs('tmp', exist_ok=True)
    train_df.to_csv('tmp/train_split.csv', index=False)
    val_df.to_csv('tmp/val_split.csv', index=False)

    shared_kwargs = dict(
        series_csv    = cfg['SERIES_CSV'],
        data_dir      = cfg['DATA_DIR'],
        plane         = cfg['PLANE'],
        target_slices = cfg['TARGET_SLICES'],
        target_size   = (cfg['IMG_SIZE'], cfg['IMG_SIZE']),
        is_train      = True,
    )

    train_ds = RSNAKneeDataset(csv_file='tmp/train_split.csv', **shared_kwargs)
    val_ds   = RSNAKneeDataset(csv_file='tmp/val_split.csv',   **shared_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train studies: {len(train_ds)} | Val studies: {len(val_ds)}")
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, steps = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for volumes, targets in tqdm(loader, desc='train' if is_train else 'val', leave=False):
            volumes = volumes.to(device)
            targets = targets.to(device)

            logits = model(volumes)

            # Mask out NaN targets so they don't contribute to loss
            mask = ~targets.isnan()
            loss = criterion(logits[mask], targets[mask])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1)


def train():
    cfg = CONFIG
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])
    device = torch.device(cfg['DEVICE'])
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)

    model = RSNAKneeModel(backbone=cfg['BACKBONE'], pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['EPOCHS'])

    best_val_loss = float('inf')

    for epoch in range(1, cfg['EPOCHS'] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{cfg['EPOCHS']} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg['SAVE_PATH'])
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {cfg['SAVE_PATH']}")


if __name__ == '__main__':
    train()
