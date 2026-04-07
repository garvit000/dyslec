#!/usr/bin/env python
# coding: utf-8
# Deep learning Model of EEG Data Analysis — Dyslexia Diagnosis
# Local training script (no Kaggle / Jupyter dependencies)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — saves plots to disk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EEG_CSV     = os.path.join(BASE_DIR, 'datasets', 'eeg', 'EEG_data.csv')
DEMO_CSV    = os.path.join(BASE_DIR, 'datasets', 'eeg', 'demographic_info.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'dyslexia_model.h5')
PLOTS_DIR   = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

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

# Encode categorical columns
df['gender']    = df['gender'].replace({'M': 1, 'F': 0})
df['ethnicity'] = df['ethnicity'].replace({'Han Chinese': 0, 'Bengali': 1, 'English': 2})

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

# ── 4. Scale Features ─────────────────────────────────────────────────────────
df_sc = StandardScaler().fit_transform(df[top_fea])

# ── 5. Train / Val / Test Split ───────────────────────────────────────────────
Xtr, xte, Ytr, yte   = train_test_split(df_sc, y, random_state=108, test_size=0.27)
xtr, xval, ytr, yval = train_test_split(Xtr,   Ytr, random_state=108, test_size=0.27)
print(f"\nSplit sizes  →  train: {len(xtr)}  val: {len(xval)}  test: {len(xte)}")

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

plt.figure(figsize=(10, 4))
training[['loss', 'val_loss']].plot()
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.tight_layout()
loss_plot = os.path.join(PLOTS_DIR, 'loss.png')
plt.savefig(loss_plot)
print(f"Loss plot     → {loss_plot}")

plt.figure(figsize=(10, 4))
training[['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.tight_layout()
acc_plot = os.path.join(PLOTS_DIR, 'accuracy.png')
plt.savefig(acc_plot)
print(f"Accuracy plot → {acc_plot}")

print("\nDone ✓")
