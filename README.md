# ScriptioContinua: Word Segmentation Pipeline

This repository contains the prototype for the **"Linguist" Module** of the Epigrascan project. 

The goal of this system is to resolve *scriptio continua* (text written without spaces) by utilizing a character-level sliding-window neural network. Rather than relying on rigid dictionaries, this model learns the structural morphology of words (prefixes, roots, suffixes) to mathematically predict word boundaries and segment continuous strings.

Currently, this repository is configured as a "toy model" using English text to establish the data pipeline and training loop before transitioning to Early Brahmi transliterations.

## 📂 Project Structure

```text
ScriptioContinua/
│
├── data/                   # Local database layer
│   ├── raw/                # Unprocessed, spaced sentences (e.g., "this is a test")
│   └── processed/          # Generated training pairs with BIOES tags (spaces stripped)
│
├── models/                 # Output directory for saved model weights (*.pt files)
│
├── notebooks/              # Scratchpad for data exploration and logic testing
│   └── 01_data_exploration.ipynb  
│
├── src/                    # Core machine learning and data processing logic
│   ├── __init__.py
│   ├── data_pipeline.py    # Converts raw sentences into BIOES tagged arrays
│   ├── dataset.py          # PyTorch Dataset class for batching and sliding windows
│   ├── model.py            # Neural network architecture (e.g., BiLSTM / 1D CNN)
│   ├── train.py            # The training loop (loss calculation, backpropagation)
│   └── predict.py          # Inference logic for segmenting new strings
│
├── main.py                 # The entry point to run the pipeline or test predictions
├── requirements.txt        # Python dependencies
├── .gitignore              
└── README.md               # You are here