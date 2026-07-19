# Pneumonia Detection from Chest X-Rays

A convolutional neural network built from scratch to detect pneumonia from chest X-ray images. No transfer learning, no pre-trained weights — just a custom CNN trained end-to-end on the Kaggle Chest X-Ray dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Design Decisions](#design-decisions)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## About the Project

Pneumonia is one of the leading causes of death worldwide, especially in children under 5. Early and accurate diagnosis from chest X-rays can significantly improve patient outcomes. This project builds a deep learning model that automates this diagnosis, classifying X-ray images as Normal or Pneumonia with high accuracy.

The entire CNN is built from scratch using TensorFlow/Keras, demonstrating a solid understanding of convolutional neural network fundamentals without relying on pre-trained architectures.

---

## Dataset

Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle

| Split      | Normal | Pneumonia | Total |
|------------|--------|-----------|-------|
| Training   | 1,341  | 3,875     | 5,216 |
| Test       | 234    | 390       | 624   |
| Validation | 8      | 8         | 16    |

Key things to note:
- The dataset is imbalanced — Pneumonia images outnumber Normal by roughly 3:1 in the training set. This is handled using class weight balancing during training.
- The validation set is tiny (only 16 images). Because of this, the test set is used to monitor training progress, and the validation set is reserved for the final unbiased evaluation only.
- All images are grayscale JPEGs of varying resolutions, resized to 150x150 for the model.

---

## Model Architecture

A 4-block CNN with progressively increasing filter depth, followed by a fully connected classification head.

```
Input (150x150x3)
    |
    +-- Conv Block 1:  Conv2D(32) -> BatchNorm -> MaxPool    ->  74x74x32
    +-- Conv Block 2:  Conv2D(64) -> BatchNorm -> MaxPool    ->  36x36x64
    +-- Conv Block 3:  Conv2D(128) -> BatchNorm -> MaxPool   ->  17x17x128
    +-- Conv Block 4:  Conv2D(128) -> BatchNorm -> MaxPool   ->   7x7x128
    |
    +-- Flatten         -> 6,272 features
    +-- Dense(512)      -> ReLU activation
    +-- Dropout(0.5)    -> Regularisation
    +-- Dense(1)        -> Sigmoid (0 = Normal, 1 = Pneumonia)
```

| Layer Type          | Purpose |
|---------------------|---------|
| Conv2D (3x3)        | Scans the image with small filters to detect visual features at increasing complexity — from simple edges (Block 1) to abstract lung patterns (Block 4) |
| BatchNormalization  | Normalises each layer's output to prevent gradients from exploding or vanishing, which stabilises and speeds up training |
| MaxPooling2D (2x2)  | Halves spatial dimensions after each block, reducing computation while keeping the most important features |
| Flatten             | Converts the final 2D feature maps into a 1D vector that the Dense layers can process |
| Dense (512)         | Fully connected layer that combines all extracted features to make the final classification decision |
| Dropout (0.5)       | Randomly disables 50% of neurons during each training step, forcing the network to learn redundant representations and preventing overfitting |
| Sigmoid             | Outputs a single probability between 0 and 1 for binary classification |

---

## Design Decisions

### Why 150x150 image size?
Full-resolution X-rays are large (1000+ pixels). Resizing to 150x150 keeps enough anatomical detail for the model while making training significantly faster. Going larger (like 224x224) would increase training time with diminishing returns for a 4-layer CNN.

### Why progressive filter sizes (32 -> 64 -> 128 -> 128)?
This mirrors how CNNs learn hierarchically:
- 32 filters detect simple edges, gradients, and basic textures
- 64 filters combine edges into recognisable shapes like rib outlines and tissue boundaries
- 128 filters capture larger patterns such as lung field structure and opacity regions
- 128 filters (again) extract the final abstract, high-level features that distinguish healthy lungs from pneumonia

Doubling filters compensates for the spatial information lost through pooling — fewer pixels, but richer features per pixel.

### Why BatchNormalization after every Conv layer?
Without it, deeper layers receive inputs with wildly varying distributions as weights update. BatchNorm standardises these distributions, which:
- Allows the model to use a reasonable learning rate without training instability
- Speeds up convergence (fewer epochs to reach good accuracy)
- Acts as a mild regulariser, complementing Dropout

### Why a learning rate of 0.0001?
The default Adam learning rate (0.001) can be too aggressive for medical imaging tasks, causing the model to overshoot good solutions. A lower rate of 0.0001 makes smaller, more precise weight updates — slower to converge but more stable and less likely to miss the optimal solution.

### Why class weight balancing?
The training set has roughly 3x more Pneumonia images than Normal. Without intervention, the model could learn a shortcut: predict "Pneumonia" for everything and still achieve about 75% accuracy. Class weights inversely scale with frequency, so the model gets penalised more heavily for misclassifying the minority class (Normal), forcing it to genuinely learn both conditions.

### Why use the test set as validation during training?
The official validation split has only 16 images. If used to track training progress, the accuracy would swing wildly between epochs (a single misclassification = 6.25% accuracy drop). The 624-image test set provides much smoother, more reliable training curves. The tiny val set is reserved for one final evaluation after training is complete.

### Why data augmentation on training data only?
Augmentation (shear, zoom, flip) artificially increases training variety so the model sees slightly different versions of each image every epoch. This teaches it to focus on the actual medical features rather than memorising specific pixel arrangements. Test and validation data must remain clean and undistorted to give an honest accuracy reading.

### Why Dropout at 0.5?
50% dropout is aggressive but effective for a model of this size. With 512 neurons in the dense layer connected to 6,272 flattened features, there are millions of parameters that could easily memorise the training data. Dropout forces the network to distribute its knowledge across many neurons rather than relying on a few, resulting in better generalisation to unseen X-rays.

---

## How to Run

### On Kaggle (Recommended)
1. Create a new Kaggle Notebook
2. Add the [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset
3. Enable GPU under Settings -> Accelerator
4. Copy the contents of `model.py` into a notebook cell and run

### Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/xray-cnn-classifier.git
cd xray-cnn-classifier

# Install dependencies
pip install -r requirements.txt

# Update the dataset path in model.py, then run
python model.py
```

Note: You'll need to download the dataset from Kaggle and update the `base_dir` path in `model.py` to point to your local copy.

---

## Results

<!-- Add your training curves and accuracy screenshots here -->
<!-- ![Training Curves](results/training_curves.png) -->
<!-- ![Confusion Matrix](results/confusion_matrix.png) -->

| Metric              | Value |
|---------------------|-------|
| Validation Accuracy | —     |
| Validation Loss     | —     |

Results will be updated after training is complete.

---

## Future Improvements

Potential enhancements that could improve the model's performance and code quality:

- EarlyStopping — Automatically stop training when performance plateaus, saving time and preventing overfitting
- ReduceLROnPlateau — Dynamically reduce learning rate when training stalls for fine-grained convergence
- ModelCheckpoint — Save the best model during training instead of the last one
- Confusion Matrix and Classification Report — Go beyond accuracy with precision, recall, and F1 score
- Precision and Recall metrics — Critical for medical imaging where false negatives are dangerous
- GlobalAveragePooling2D — Replace Flatten to reduce parameters by roughly 50x and improve generalisation
- tf.data pipeline — Replace deprecated ImageDataGenerator with modern, faster data loading
- Reproducibility seeds — Set random seeds for consistent, reproducible results
- Additional augmentation — Add rotation and contrast variation for better robustness

---

## Project Structure

```
Pneumonia_CNN/
├── model.py              # CNN architecture, training, and evaluation
├── README.md             # Project documentation (this file)
├── requirements.txt      # Python dependencies
├── .gitignore            # Files excluded from version control
├── LICENSE               # MIT License
└── results/              # Training curves, confusion matrix, metrics
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
