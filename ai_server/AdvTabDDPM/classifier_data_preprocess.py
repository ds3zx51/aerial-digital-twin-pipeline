import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # add
import pickle
from collections import defaultdict

import numpy as np
import torch

# -------------------------------------------------------------------
# Dataset Generation
# -------------------------------------------------------------------
def gen_slice_dataset_from_numpy(
    data_dir,
    use_diffusion_data=False,
    slice_len=4,
    drop_colnames=[]
):
    """
    Generate dataset from preprocessed numpy arrays (numerical + categorical).
    
    Steps:
      1. Load train/val split data.
      2. Preprocess (remove excluded columns, adjust timestamp).
      3. Slice sequences per label (non-sequential).
      4. Normalize features using training data only.
      5. Save slice-to-original mapping (for boundary analysis).
    
    Args:
        data_dir (str): Path to directory containing numpy arrays.
        use_diffusion_data (bool): Skip validation set if True.
        slice_len (int): Length of each slice.
        drop_colnames (list): Columns to exclude (default: empty).

    Returns:
        dataset (dict): Contains train/valid samples (norm + no_norm) and labels.
        norm_param (dict): Normalization parameters and metadata.
    """
    print(f"[INFO] Loading data from {data_dir}")

    # --------------------------------------------------------------
    # Step 1: Data Loading
    # --------------------------------------------------------------
    def load_split(split):
        X_cat = np.load(os.path.join(data_dir, f'X_cat_{split}.npy'), allow_pickle=True)
        X_num = np.load(os.path.join(data_dir, f'X_num_{split}.npy'), allow_pickle=True)
        y     = np.load(os.path.join(data_dir, f'y_{split}.npy'), allow_pickle=True)
        return X_cat, X_num, y

    original_col_names = [
        'Timestamp', 'num_ues', 'slice_prb', 'dl_mcs', 'dl_n_samples',
        'dl_buffer [bytes]', 'tx_brate downlink [Mbps]', 'tx_pkts downlink',
        'dl_cqi', 'ul_mcs', 'ul_n_samples', 'ul_buffer [bytes]',
        'rx_brate uplink [Mbps]', 'rx_pkts uplink', 'rx_errors uplink (%)',
        'ul_sinr', 'phr', 'sum_requested_prbs', 'sum_granted_prbs',
        'ul_turbo_iters', 'RNTI', 'slice_id', 'scheduling_policy'
    ]
    exclude_colnames = ['slice_id']
    exclude_cols_ix = [original_col_names.index(name) for name in exclude_colnames]
    raw_cols_names = np.array(original_col_names)

    # --------------------------------------------------------------
    # Step 2: Preprocess
    # --------------------------------------------------------------
    def preprocess(X_num, X_cat):
        X_num = X_num.astype(np.float64)
        X_cat = X_cat.astype(np.float64)
        X = np.hstack([X_num, X_cat])
        X_filtered = np.delete(X, exclude_cols_ix, axis=1)

        # Normalize timestamp by subtracting first value
        if raw_cols_names[0] == 'Timestamp':
            X_filtered[:, 0] -= X_filtered[0, 0].copy()
        return X_filtered
    
    # --------------------------------------------------------------
    # Step 3: Slice by Label
    # --------------------------------------------------------------
    def group_by_label(X, y):
        label_to_samples = defaultdict(list)
        for xi, yi in zip(X, y):
            label_to_samples[int(yi)].append(xi)
        return label_to_samples

    def slice_non_sequential_by_label(X, y, slice_len):
        label_to_samples = group_by_label(X, y)
        X_slices, y_slices = [], []
        slice_to_original_map = []

        for label, samples in label_to_samples.items():
            samples = np.array(samples)
            if len(samples) < slice_len:
                continue

            for i in range(len(samples) - slice_len + 1):
                X_slices.append(samples[i:i+slice_len])
                y_slices.append(label)
                # Mapping to original index for boundary inputs
                slice_to_original_map.append((label, i))
        
        return (
            np.stack(X_slices),
            np.array(y_slices),
            slice_to_original_map
        )

    # --------------------------------------------------------------
    # Step 4: Train Split
    # --------------------------------------------------------------
    X_cat_train, X_num_train, y_train = load_split('train')
    X_train = preprocess(X_num_train, X_cat_train)
    D = X_train.shape[1]

    X_sliced_train, y_sliced_train, slice_map_train = slice_non_sequential_by_label(
        X_train, y_train, slice_len
    )

    # Normalization (train only)
    feat_min = X_sliced_train.min(axis=(0, 1))
    feat_max = X_sliced_train.max(axis=(0, 1))
    normalize = lambda X: (X - feat_min) / (feat_max - feat_min + 1e-8)

    dataset = {}
    dataset['train'] = create_split(
        samples=X_sliced_train,
        norm_samples=normalize(X_sliced_train),
        labels=y_sliced_train
    )

    # Save slice-to-original mapping
    with open(os.path.join(data_dir, 'slice_to_original_map.pkl'), 'wb') as f:
        pickle.dump(slice_map_train, f)

    # --------------------------------------------------------------
    # Step 5: Validation Split
    # --------------------------------------------------------------
    if not use_diffusion_data:
        X_cat_val, X_num_val, y_val = load_split('val')
        X_val = preprocess(X_num_val, X_cat_val)
        X_sliced_val, y_sliced_val, _ = slice_non_sequential_by_label(X_val, y_val, slice_len)
        dataset['valid'] = create_split(
            samples=X_sliced_val,
            norm_samples=normalize(X_sliced_val),
            labels=y_sliced_val
        )

    # --------------------------------------------------------------
    # Step 6: Norm Param Metadata
    # --------------------------------------------------------------
    norm_param = {
        i: {
            'name': raw_cols_names[i],
            'min': float(feat_min[i]),
            'max': float(feat_max[i])
        }
        for i in range(D)
    }
    norm_param['info'] = {
        'raw_cols_names': raw_cols_names,
        'numerical_dim': D,
        'exclude_cols_ix': exclude_cols_ix
    }

    return dataset, norm_param

# -------------------------------------------------------------------
# Utility: Create Split Dict
# -------------------------------------------------------------------
def create_split(samples, norm_samples, labels):
    """Utility to pack split data into dict format."""
    return {
        'samples': {
            'no_norm': torch.as_tensor(
                samples,
                dtype=torch.float32
            ),
            'norm': torch.as_tensor(
                norm_samples,
                dtype=torch.float32
            )
        },
        'labels': torch.as_tensor(
            labels,
            dtype=torch.long
        )
    }

def safe_pickle_dump(filepath, obj, overwrite=True):
    """
    Save Python object to pickle file safely.

    Args:
        filepath (str): Path to save pickle file.
        obj (object): Python object to save.
        overwrite (bool): If False and file exists, skip saving.

    Returns:
        bool: True if saved, False if skipped.
    """
    if os.path.isfile(filepath):
        if overwrite:
            print(f"[INFO] {filepath} already exists. Overwriting...")
        else:
            print(f"[INFO] {filepath} already exists. Skipping save.")
            return False

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved object to {filepath}")
    return True