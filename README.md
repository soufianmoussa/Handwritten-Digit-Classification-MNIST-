# MNIST Digit Classification

This project implements a neural network to classify handwritten digits (0-9) using the standard MNIST dataset (28x28).

## Architecture
- **Input**: 784 units (28x28 flattened image)
- **Hidden**: 25 units
- **Output**: 10 units
- **Dataset**: MNIST (60,000 training images)
- **Preprocessing**: Robust cropping and centering to match MNIST distribution.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model (Migration)
This script will download MNIST, augment the data, train the network, and save `model_weights.npz`.

```bash
python train.py
```

### 2. Run the App
Launch the Streamlit interface:

```bash
python -m streamlit run app.py
```

- Draw a digit on the canvas.
- The app will show the prediction, confidence, and the "seen" processed image.

## Files
- `mnist_loader.py`: Downloads and parses MNIST data.
- `train.py`: Training script with SciPy optimization and Data Augmentation.
- `preprocess.py`: Image processing (28x28 resizing, CoM centering).
- `model.py`: Prediction logic.
- `app.py`: Streamlit application.
