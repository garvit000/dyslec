#!/usr/bin/env python
# coding: utf-8
# Deep learning Model of EEG Data Analysis — Dyslexia Diagnosis
# Local training script (no Kaggle / Jupyter dependencies)

import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — saves plots to disk
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Windows consoles default to cp1252, which cannot encode the arrows/checkmarks
# used below; fall back to replacing them rather than crashing mid-run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EEG_CSV     = os.path.join(BASE_DIR, 'datasets', 'eeg', 'EEG_data.csv')
DEMO_CSV    = os.path.join(BASE_DIR, 'datasets', 'eeg', 'demographic_info.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'dyslexia_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.json')
PLOTS_DIR   = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

for path, name in ((EEG_CSV, 'EEG_data.csv'), (DEMO_CSV, 'demographic_info.csv')):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing dataset file: {path}\n"
            f"Download {name} from https://www.kaggle.com/datasets/wanghaohan/confused-eeg "
            f"and extract it to model/datasets/eeg/."
        )

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading datasets...")
df   = pd.read_csv(EEG_CSV)
data = pd.read_csv(DEMO_CSV)
print(f"  EEG data      : {df.shape}")
print(f"  Demographic   : {data.shape}")

# ── 2. Merge & Preprocess ─────────────────────────────────────────────────────
data = data.rename(columns={
    'subject ID': 'SubjectID',
    ' gender'   : 'gender',
    ' age'      : 'age',
    ' ethnicity': 'ethnicity'
})
df = df.merge(data, how='inner', on='SubjectID')
print(f"  Merged shape  : {df.shape}")

# Encode categorical columns. .map() (rather than .replace()) turns any
# unexpected category into NaN so it shows up in the null check below.
df['gender']    = df['gender'].map({'M': 1, 'F': 0})
df['ethnicity'] = df['ethnicity'].map({'Han Chinese': 0, 'Bengali': 1, 'English': 2})

# Check for nulls
null_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
if null_cols:
    print(f"  Columns with nulls: {null_cols}")
else:
    print("  No null values found ✓")

# ── 3. Feature Selection via Mutual Information ───────────────────────────────
print("\nRunning mutual information feature selection...")
y        = pd.get_dummies(df['user-definedlabeln'])
mi_score = mutual_info_classif(df.drop('user-definedlabeln', axis=1), df['user-definedlabeln'])
mi_score = pd.Series(mi_score, index=df.drop('user-definedlabeln', axis=1).columns)
mi_score = (mi_score * 100).sort_values(ascending=False)
print(mi_score.head(14))

top_fea = ['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
           'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']

# ── 4. Train / Val / Test Split ───────────────────────────────────────────────
X = df[top_fea]
Xtr, xte, Ytr, yte   = train_test_split(X,   y,   random_state=108, test_size=0.27)
xtr, xval, ytr, yval = train_test_split(Xtr, Ytr, random_state=108, test_size=0.27)
print(f"\nSplit sizes  →  train: {len(xtr)}  val: {len(xval)}  test: {len(xte)}")

# ── 5. Scale Features ─────────────────────────────────────────────────────────
# Fit on the training split only, then apply to val/test. Fitting on the full
# dataset would leak val/test statistics into training.
scaler = StandardScaler().fit(xtr)
xtr, xval, xte = (scaler.transform(part) for part in (xtr, xval, xte))

joblib.dump(scaler, SCALER_PATH)
with open(FEATURES_PATH, 'w', encoding='utf-8') as fh:
    json.dump(top_fea, fh, indent=2)
print(f"Scaler saved  → {SCALER_PATH}")

# ── 6. Build Model ────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = keras.Sequential([
    layers.Dense(64,  input_shape=(14,), activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.27),
    layers.Dense(124, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.3),
    layers.Dense(248, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.32),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.27),
    layers.Dense(664, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.32),
    layers.Dense(264, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.27),
    layers.Dense(124, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.3),
    layers.Dense(2,   activation='sigmoid')
])
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\nTraining...")
early_stop = callbacks.EarlyStopping(patience=20, min_delta=0.0001, restore_best_weights=True)
history = model.fit(
    xtr, ytr,
    validation_data=(xval, yval),
    batch_size=28,
    epochs=150,
    callbacks=[early_stop],
    verbose=1
)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
loss, acc = model.evaluate(xte, yte, verbose=0)
print(f"  Test loss    : {loss:.4f}")
print(f"  Test accuracy: {acc:.4f}")

# ── 9. Save Model ─────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── 10. Plot Training History ─────────────────────────────────────────────────
training = pd.DataFrame(history.history)

# pandas' .plot() creates its own figure, so draw onto an explicit axis —
# calling plt.figure() first would leave the styled figure blank.
for cols, title, ylabel, fname in (
    (['loss', 'val_loss'],         'Loss',     'Loss',     'loss.png'),
    (['accuracy', 'val_accuracy'], 'Accuracy', 'Accuracy', 'accuracy.png'),
):
    fig, ax = plt.subplots(figsize=(10, 4))
    training[cols].plot(ax=ax)
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"{title} plot → {out_path}")

print("\nDone ✓")
