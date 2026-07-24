# Dyslec

## What is Dyslec
Dyslec is an open-source project made for analysis of our research on *Advancing Early Detection of Dyslexia by Leveraging AI and Neuroimaging for Enhanced Diagnosis*.

The working pipeline in this repository is the **EEG confusion classifier**: a fully-connected Keras network trained on the Confused-EEG dataset. The eye-tracking, handwriting, and fMRI directories hold dataset documentation and exploratory scripts for planned work.

## Datasets
Datasets are **not** included in this repository (they are large and separately licensed). Download them and extract into the paths below:

| Dataset | Source | Extract to |
| --- | --- | --- |
| EEG Confusion Data | [Kaggle](https://www.kaggle.com/datasets/wanghaohan/confused-eeg) | `model/datasets/eeg/` |
| ETDD70 Eye-Tracking | [Zenodo](https://doi.org/10.5281/zenodo.13332134) | `model/datasets/eye-tracking/` |

The EEG pipeline requires exactly these two files:
```text
model/datasets/eeg/EEG_data.csv
model/datasets/eeg/demographic_info.csv
```

## Getting Started

### 1. Install dependencies
```bash
cd model
pip install -r requirements.txt
```
`nilearn`/`nibabel` are only needed for the optional fMRI viewer, and `pytest` only for the tests.

### 2. Run inference
A pre-trained model (`dyslexia_model.h5`) is committed, so you can predict without training. From the `model` directory:
```bash
python run.py                  # score a random sample from the dataset
python run.py --sample 328     # score a specific row
python run.py --seed 42        # reproducible random sample
python run.py --features 6 51 14493 1071485 35687 180000 8695 26390 41 21219 7 75667 11 25
```
`--features` takes the 14 raw values in the order printed by the script, and is the only mode that works without the dataset present.

> **Note**: `run.py` works around Keras version drift (the `quantization_config` and `renorm` arguments that older saved models carry) by stripping those arguments at load time, so it is not tied to one TensorFlow version.

### 3. Train from scratch
```bash
python train.py
```
This writes:
- `dyslexia_model.h5` — the trained network
- `scaler.joblib` and `features.json` — the fitted `StandardScaler` and feature order, which `run.py` loads so inference scales inputs exactly as training did
- `plots/accuracy.png` and `plots/loss.png` — training curves

The scaler is fit on the **training split only**, so validation and test statistics do not leak into training. Current held-out test accuracy is ~94.5%.

If you train with an older checkpoint and no `scaler.joblib` is present, `run.py` warns and refits a scaler on the full dataset as a fallback — re-run `train.py` to remove the warning.

### 4. Run the tests
```bash
python -m pytest test_model.py
```
Tests skip automatically when the model or dataset is missing.

## Repository layout
```text
model/
  train.py        # trains the EEG classifier, saves model + scaler + plots
  run.py          # inference CLI
  test_model.py   # smoke tests for the pipeline
  view_fmri.py    # optional standalone fMRI scan viewer
  eeg-analysis-of-confusion-for-dyslexia-diagnosis.{py,ipynb}
                  # original exploratory Kaggle notebook and its export
  datasets/       # dataset docs; data itself is gitignored
```

## Caveats
- `SubjectID` and `VideoID` are used as input features. They carry real signal on this dataset but are identifiers, not physiological measurements, so accuracy here will not transfer directly to unseen subjects or videos.
- The exploratory notebook (`eeg-analysis-of-confusion-for-dyslexia-diagnosis.py`) is kept for reference and reflects the original Kaggle environment; `train.py` is the maintained path.
