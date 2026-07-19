"""
Pneumonia Detection from Chest X-Rays — Custom CNN
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


# CONFIGURATION 
base_dir = '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

print("--> Environment Ready.")
print(f"--> Checking GPU: {tf.config.list_physical_devices('GPU')}")


# DATA AUGMENTATION & LOADING 
# Training data gets augmented to increase variety and reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalise pixels to [0, 1]
    shear_range=0.2,         # Slight slanting
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True     # Mirror images
)

# Test/Val data only gets rescaled — no distortion on evaluation data
test_val_datagen = ImageDataGenerator(rescale=1./255)

print("Loading Datasets...")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Test set used to track progress during training as the val set is too small
test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Validation set reserved for the final evaluation
val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False     # Keep order intact for clear prediction mapping
)


# MODEL ARCHITECTURE
def build_custom_model():
    """4-block CNN for binary chest X-ray classification (Normal vs Pneumonia)."""

    model = Sequential([
        # Block 1 — 32 filters for basic edge detection
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 2 — 64 filters for texture recognition
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 3 — 128 filters for pattern detection
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 4 — 128 filters for high-level feature extraction
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Classification head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),                        # 50% dropout to prevent overfitting
        Dense(1, activation='sigmoid')       # Binary output: 0=Normal, 1=Pneumonia
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # low learning rate for stability
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


model = build_custom_model()
print("Model Built Successfully.")
model.summary()


# CLASS WEIGHT BALANCING
# Dataset is imbalanced (~3x more Pneumonia than Normal), so we assign
# higher weights to the minority class to prevent biased predictions
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
weights_dict = dict(enumerate(weights))
print(f"Balancing Weights: {weights_dict}")


# TRAINING
print("\n Starting Training...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=15,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    class_weight=weights_dict
)

print("Training Complete.")
model.save('final_chest_xray_model.h5')


#TRAINING CURVES
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')

plt.show()


#FINAL VALIDATION
print("Performing Final Validation on Dedicated Validation Set...")

val_loss, val_accuracy = model.evaluate(val_generator)

print("-" * 30)
print(f"FINAL VALIDATION ACCURACY: {val_accuracy * 100:.2f}%")
print(f"FINAL VALIDATION LOSS:     {val_loss:.4f}")
print("-" * 30)


#SAMPLE PREDICTION
imgs, labels = next(val_generator)
preds = model.predict(imgs)

print(f"\nSample Prediction (0=Normal, 1=Pneumonia):")
print(f"Actual:    {int(labels[0])}")
print(f"Predicted: {int(round(preds[0][0]))}")
