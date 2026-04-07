# Dyslec
## What is dyslec
Dyslec is an open-source project made for analysis of our research on *Advancing Early Detection of Dyslexia by Leveraging AI and Neuroimaging for Enhanced Diagnosis*. <br>
Datasets are not uploaded here but they can be found by accessing the following links: <br>

- [EEG Confusion Data (Kaggle)](https://www.kaggle.com/datasets/wanghaohan/confused-eeg) (Extract to `model/datasets/eeg/`)

## Getting Started

### 1. Install Dependencies
You can use `pip` to install the latest requirements. Navigate to the `model` folder and run:
```bash
cd model
pip install -r requirements.txt
```

### 2. File Placement
The model requires the EEG dataset to generate a feature scaler dynamically. Ensure you have downloaded the datasets and placed them in the appropriate structure:
```text
model/datasets/eeg/EEG_data.csv
model/datasets/eeg/demographic_info.csv
```

### 3. Running Inference (Making Predictions)
Our repository includes a pre-compiled model compatible script. Once datasets and dependencies are present, execute the following script from the `model` directory to feed a random sample into the Neural Net and read its predictions:
```bash
python run.py
```
> **Note**: Our `run.py` logic natively circumvents Keras Versioning issues (specifically `quantization_config` and `renorm` incompatibilities on old TF model imports) by dynamically overriding incompatible hyperparameters. It's designed to run independently of your local TensorFlow version differences!

### 4. Training the Model from Scratch
Want to tweak the Neural Network architecture or train on new data? Run the script to retrain the sequential fully-connected network locally (avoiding Jupyter artifacts):
```bash
python train.py
```
This will generate custom evaluation plots (`accuracy.png`, `loss.png`) in the `./model/plots/` folder so you can visualize the training bounds and then save output to `dyslexia_model.h5`.
