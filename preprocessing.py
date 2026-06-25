"""
Preprocessing utilities for Task 2: Regression Pipeline
Includes OOF target encoding and imputation.
"""

import numpy as np
from sklearn.model_selection import KFold
from config import IMPUTE_COLS, RANDOM_STATE, N_FOLDS


def apply_oof_target_encoding(train, test, y, n_folds=N_FOLDS):
    """Apply out-of-fold target encoding for 'channel' column to prevent data leakage."""
    kf_te = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    train['channel_target_enc'] = np.nan

    for tr_idx, val_idx in kf_te.split(train):
        tr_channels = train.iloc[tr_idx]['channel']
        tr_target = y.iloc[tr_idx]

        mean_enc = tr_target.groupby(tr_channels).mean()
        train.loc[val_idx, 'channel_target_enc'] = train.iloc[val_idx]['channel'].map(mean_enc)

    full_mean_enc = y.groupby(train['channel']).mean()
    test['channel_target_enc'] = test['channel'].map(full_mean_enc)

    global_mean_log_shares = y.mean()
    train['channel_target_enc'] = train['channel_target_enc'].fillna(global_mean_log_shares)
    test['channel_target_enc'] = test['channel_target_enc'].fillna(global_mean_log_shares)

    return train, test


def fit_imputation_medians(train):
    """Calculate median values from training data for imputation."""
    return {col: train[col].median() for col in IMPUTE_COLS}


def apply_imputation(df, median_fills):
    """Apply imputation using pre-computed medians."""
    df = df.copy()
    for col, val in median_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df
