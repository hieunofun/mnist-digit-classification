# MNIST Handwritten Digit Classification

Train and evaluate convolutional neural networks for classifying handwritten digits (0–9) using **PyTorch**. This repository contains the recovered CNN implementation of the MNIST project.

## Features

- A compact `MNISTCNN` with convolution, batch normalization, dropout and a dense classifier.
- A configurable deeper `StackedCNN` for architecture experiments.
- MNIST normalization, mini-batch training and evaluation.
- Confusion matrices, learning curves and visual inspection of misclassified digits.
- CPU, CUDA or Apple MPS device selection when available.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place `mnist.npz` in the repository directory. Its arrays must be `x_train`, `y_train`, `x_test` and `y_test` in the standard Keras MNIST format. Alternatively, install TensorFlow to enable the script's Keras dataset download fallback:

```bash
pip install tensorflow
python mnist_cnn_from_scratch.py --epochs 10 --batch-size 128 --output-dir outputs
```

For an existing dataset file, TensorFlow is optional:

```bash
python mnist_cnn_from_scratch.py --data mnist.npz --epochs 10
python mnist_cnn_from_scratch.py --help
python mnist_cnn_from_scratch.py --deep-experiment --deep-epochs 3
```

Training results are written to `outputs/`, which is excluded from Git. The published source is the CNN experiment; the earlier MLP notebook and its reported 98.93% result are not included in this recovered source. No accuracy figure is presented here as a newly reproduced measurement.

## Technology

Python, PyTorch, NumPy, Matplotlib and scikit-learn. TensorFlow/Keras is used only for the optional MNIST data-loading fallback.
