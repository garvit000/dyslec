import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import argparse

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EEG_CSV     = os.path.join(BASE_DIR, 'datasets', 'eeg', 'EEG_data.csv')
DEMO_CSV    = os.path.join(BASE_DIR, 'datasets', 'eeg', 'demographic_info.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'dyslexia_model.h5')

def prepare_scaler():
    """Reads the dataset just to fit the StandardScaler as it was during training."""
    if not os.path.exists(EEG_CSV):
        print("Data files not found to prepare standard scaler.")
        return None
        
    df   = pd.read_csv(EEG_CSV)
    data = pd.read_csv(DEMO_CSV)
    
    data = data.rename(columns={
        'subject ID': 'SubjectID',
        ' gender'   : 'gender',
        ' age'      : 'age',
        ' ethnicity': 'ethnicity'
    })
    df = df.merge(data, how='inner', on='SubjectID')
    
    top_fea = ['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
               'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']
               
    scaler = StandardScaler()
    scaler.fit(df[top_fea])
    return scaler, df, top_fea

def run_inference(sample_data, scaler, model, top_fea):
    """
    Run inference on a given sample
    sample_data: list or numpy array of the 14 feature values
    """
    # Create pandas dataframe to retain feature names when transforming
    sample_df = pd.DataFrame([sample_data], columns=top_fea)
    sample_scaled = scaler.transform(sample_df)
    
    # Predict
    prediction = model.predict(sample_scaled, verbose=0)
    
    # Class probabilities usually come out as [prob_class_0, prob_class_1] because of final dense layer with 2 units & sigmoid
    return prediction[0]

class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}! Please train the model first.")
        return

    print(f"Loading EEG Keras Model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'Dense': CustomDense,
        'BatchNormalization': CustomBatchNormalization
    })
    
    print("Preparing feature scaler...")
    scaler, df, top_fea = prepare_scaler()
    
    if scaler is None:
        return
        
    print("\n" + "="*60)
    print("Dyslexia EEG Model Inference")
    print("="*60)
    print(f"The model requires the following {len(top_fea)} features as input:")
    
    # Display the required features
    for i, feature in enumerate(top_fea):
        print(f"{i+1}. {feature}")

    print("\n--- Example: Inference on a random sample from the dataset ---")
    
    # Let's pick a random sample from our dataset
    random_index = np.random.randint(0, len(df))
    sample = df.iloc[random_index]
    
    sample_features = sample[top_fea].values
    
    print("\nSample Inputs:")
    for feature_name, value in zip(top_fea, sample_features):
         print(f"  {feature_name}: {value}")
         
    actual_label = sample['user-definedlabeln']
    
    prediction = run_inference(sample_features, scaler, model, top_fea)
    
    print("\n--- Inference Results ---")
    print(f"Raw Model Output (Sigmoid Probabilities): {prediction}")
    predicted_class = np.argmax(prediction)
    print(f"Predicted Class: {predicted_class}")
    print(f"Actual Expected Class: {actual_label}")
    
    if predicted_class == actual_label:
        print("\n✅ The model made the correct prediction!")
    else:
        print("\n❌ The model made an incorrect prediction on this sample.")

if __name__ == "__main__":
    main()