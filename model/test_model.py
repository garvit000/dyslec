"""Smoke tests for the dyslexia EEG model pipeline.

Run with:  python -m pytest test_model.py
These are skipped automatically when the model or dataset is not present.
"""

import json
import os

import numpy as np
import pytest

import run

pytestmark = pytest.mark.skipif(
    not os.path.exists(run.MODEL_PATH),
    reason="dyslexia_model.h5 not present — run train.py first",
)


@pytest.fixture(scope="module")
def features():
    return run.load_features()


@pytest.fixture(scope="module")
def model():
    return run.tf.keras.models.load_model(run.MODEL_PATH, custom_objects={
        'Dense': run.CustomDense,
        'BatchNormalization': run.CustomBatchNormalization,
    })


def test_features_match_model_input(features, model):
    assert len(features) == 14
    assert model.input_shape[-1] == len(features)


def test_saved_features_match_default():
    if not os.path.exists(run.FEATURES_PATH):
        pytest.skip("features.json not present — run train.py first")
    with open(run.FEATURES_PATH, encoding='utf-8') as fh:
        assert json.load(fh) == run.DEFAULT_FEATURES


def test_inference_returns_probabilities(features, model):
    df = run.load_dataset(features)
    if df is None:
        pytest.skip("EEG dataset not present")

    scaler = run.load_scaler(df, features)
    sample = df.iloc[0][features].values

    prediction = run.run_inference(sample, scaler, model, features)

    assert prediction.shape == (2,)
    assert np.all((prediction >= 0) & (prediction <= 1))


def test_inference_is_deterministic(features, model):
    df = run.load_dataset(features)
    if df is None:
        pytest.skip("EEG dataset not present")

    scaler = run.load_scaler(df, features)
    sample = df.iloc[0][features].values

    first = run.run_inference(sample, scaler, model, features)
    second = run.run_inference(sample, scaler, model, features)

    np.testing.assert_allclose(first, second)
