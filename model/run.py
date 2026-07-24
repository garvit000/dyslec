"""Run inference with the trained dyslexia EEG model.

By default this picks a random sample from the EEG dataset and reports the
model's prediction against the known label. Pass --sample to score a specific
row, or --features to score 14 values supplied on the command line.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Suppress TF's C++ logging before it is imported, otherwise the banner still prints.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import warnings
warnings.filterwarnings('ignore')

import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Windows consoles default to cp1252 and cannot encode every character printed
# below; fall back to replacing them rather than crashing mid-run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
EEG_CSV       = os.path.join(BASE_DIR, 'datasets', 'eeg', 'EEG_data.csv')
DEMO_CSV      = os.path.join(BASE_DIR, 'datasets', 'eeg', 'demographic_info.csv')
MODEL_PATH    = os.path.join(BASE_DIR, 'dyslexia_model.h5')
SCALER_PATH   = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.json')

DEFAULT_FEATURES = ['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta',
                    'Beta1', 'Alpha1', 'Mediation', 'Gamma2', 'SubjectID',
                    'Beta2', 'Raw', 'age']


def load_features():
    """Feature order the model was trained on, as recorded by train.py."""
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    return DEFAULT_FEATURES


def load_dataset(top_fea):
    """Load and merge the EEG + demographic CSVs."""
    missing = [p for p in (EEG_CSV, DEMO_CSV) if not os.path.exists(p)]
    if missing:
        print("Dataset files not found:")
        for path in missing:
            print(f"  {path}")
        print("\nDownload the 'Confused student EEG' dataset from")
        print("https://www.kaggle.com/datasets/wanghaohan/confused-eeg")
        print("and extract it to model/datasets/eeg/.")
        return None

    df = pd.read_csv(EEG_CSV)
    demo = pd.read_csv(DEMO_CSV).rename(columns={
        'subject ID': 'SubjectID',
        ' gender'   : 'gender',
        ' age'      : 'age',
        ' ethnicity': 'ethnicity',
    })
    return df.merge(demo, how='inner', on='SubjectID')


def load_scaler(df, top_fea):
    """Load the scaler saved during training.

    Falls back to refitting on the dataset for models trained before the scaler
    was persisted; that only approximates the original fit, so warn about it.
    """
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)

    print("WARNING: scaler.joblib not found — refitting a scaler on the full "
          "dataset.\n         This approximates training-time scaling and may "
          "shift predictions.\n         Re-run train.py to generate a saved scaler.")
    if df is None:
        return None
    return StandardScaler().fit(df[top_fea])


class CustomDense(tf.keras.layers.Dense):
    """Drops `quantization_config`, absent from older Keras Dense layers."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    """Drops `renorm*` args removed in newer Keras BatchNormalization."""

    def __init__(self, *args, **kwargs):
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)


def run_inference(sample_data, scaler, model, top_fea):
    """Score a single sample given as a sequence of the 14 feature values."""
    # Use a DataFrame so feature names survive the transform and sklearn does
    # not warn about missing column names.
    sample_df = pd.DataFrame([np.asarray(sample_data, dtype=float)], columns=top_fea)
    sample_scaled = scaler.transform(sample_df)
    return model.predict(sample_scaled, verbose=0)[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sample', type=int, metavar='N',
                       help='index of the dataset row to score (default: random)')
    group.add_argument('--features', type=float, nargs=14, metavar='V',
                       help='14 raw feature values to score directly')
    parser.add_argument('--seed', type=int, help='seed for the random sample choice')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return 1

    top_fea = load_features()

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'Dense': CustomDense,
        'BatchNormalization': CustomBatchNormalization,
    })

    df = load_dataset(top_fea)
    # Explicit --features can be scored without the dataset present.
    if df is None and args.features is None:
        return 1

    scaler = load_scaler(df, top_fea)
    if scaler is None:
        return 1

    print("\n" + "=" * 60)
    print("Dyslexia EEG Model Inference")
    print("=" * 60)
    print(f"Model input features ({len(top_fea)}): {', '.join(top_fea)}")

    actual_label = None
    if args.features is not None:
        sample_features = args.features
    else:
        if args.sample is not None:
            if not 0 <= args.sample < len(df):
                print(f"--sample must be between 0 and {len(df) - 1}.")
                return 1
            index = args.sample
        else:
            rng = np.random.default_rng(args.seed)
            index = int(rng.integers(0, len(df)))

        sample = df.iloc[index]
        sample_features = sample[top_fea].values
        actual_label = sample['user-definedlabeln']
        print(f"\n--- Sample #{index} ---")

    print("\nSample inputs:")
    for feature_name, value in zip(top_fea, sample_features):
        print(f"  {feature_name}: {value}")

    prediction = run_inference(sample_features, scaler, model, top_fea)
    predicted_class = int(np.argmax(prediction))

    print("\n--- Inference Results ---")
    print(f"Raw model output (sigmoid): {prediction}")
    print(f"Predicted class: {predicted_class}")

    if actual_label is not None:
        actual_class = int(actual_label)
        print(f"Actual class   : {actual_class}")
        if predicted_class == actual_class:
            print("\n[OK] The model made the correct prediction.")
        else:
            print("\n[MISS] The model made an incorrect prediction on this sample.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
