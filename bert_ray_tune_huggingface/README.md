# NLP Model Tuning with Ray Tune -Hugging Face


# NLP Getting Started with BERT

This repository contains a script to train a BERT model for sequence classification using PyTorch, Transformers, and Ray Tune for hyperparameter tuning. It targets the Kaggle competition for NLP getting started.

## Requirements

You will need to have the following packages installed:

```bash
pip install transformers
pip install wandb
pip install torch --upgrade
pip install datasets evaluate
pip install xformers
pip install ray tune
pip install kaggle --upgrade
```

# Kaggle Setup

To download the dataset, you'll need to set up the Kaggle API credentials:

```bash
mkdir -p ~/.kaggle
mv /path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list
kaggle competitions download -c nlp-getting-started
```
Make sure you have the kaggle.json file in the appropriate directory.

# Script Description

The script provided contains the following functionalities:

Data Preprocessing: Loads the dataset, unzips, and prepares it by dropping unnecessary columns and renaming others.

Model Training Function (train_model) Definition: Contains the training loop, validation, checkpoint handling, data splitting, tokenization, transformation, data loading, model and loss initialization, optimizer initialization, and device configuration.

Hyperparameter Tuning with Ray: Utilizes Ray Tune to optimize hyperparameters such as learning rate, batch size, weight decay, etc.
Helper Functions.

The script also imports helper functions for unzipping data, creating TensorBoard callbacks, plotting loss curves, and comparing histories.

```bash
wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
```
# Running the Script
You can run the script by executing it in a Jupyter Notebook environment like Google Colab. It will train the model, evaluate it, and save checkpoints for different epochs.

# Credit for helper functions goes to [this repository](https://github.com/mrdbourke/tensorflow-deep-learning/tree/main).


