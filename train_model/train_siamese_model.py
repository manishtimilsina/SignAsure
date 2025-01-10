import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
# GENUINE_PATH = "train_model/dataset/train/genuine/"
GENUINE_PATH = "/Users/manishtimilsina/Desktop/demo/signature_verification/train_model/dataset/train/genuine/"
FORGED_PATH = "/Users/manishtimilsina/Desktop/demo/signature_verification/train_model/dataset/train/forge/"
# FORGED_PATH = "train_model/dataset/train/forge/"
IMG_HEIGHT, IMG_WIDTH = 105, 105

# Check for GPU support
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Load and preprocess images
def load_images(path):
    images = []
    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        if os.path.isfile(img_path):  # Ensure it's a file
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Skip invalid files
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img = img.astype("float32") / 255.0
                images.append(img.reshape(IMG_HEIGHT, IMG_WIDTH, 1))  # Add channel dimension
    return np.array(images)

# Load data
genuine_images = load_images(GENUINE_PATH)
forged_images = load_images(FORGED_PATH)
genuine_labels = np.ones(len(genuine_images))
forged_labels = np.zeros(len(forged_images))
X = np.concatenate([genuine_images, forged_images], axis=0)
y = np.concatenate([genuine_labels, forged_labels], axis=0)

# Generate a subset of pairs
def create_pairs_subset(images, labels, num_pairs):
    pairs, pair_labels = [], []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(images), 2, replace=False)
        pairs.append(np.stack([images[idx1], images[idx2]], axis=0))
        pair_labels.append(1 if labels[idx1] == labels[idx2] else 0)
    return np.array(pairs), np.array(pair_labels)

# Generate a limited number of pairs
NUM_PAIRS = 10000  # Adjust this based on dataset size and compute power
pairs, pair_labels = create_pairs_subset(X, y, NUM_PAIRS)

# Split data
train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    pairs, pair_labels, test_size=0.2, random_state=42
)

# Define the Siamese model
def build_siamese_model(input_shape):
    input = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation="relu")(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return Model(input, x)

# Build the model
base_network = build_siamese_model((IMG_HEIGHT, IMG_WIDTH, 1))
input_a = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
input_b = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
features_a = base_network(input_a)
features_b = base_network(input_b)

# Lambda layer to compute absolute difference
@register_keras_serializable()
def compute_distance(x):
    return tf.abs(x[0] - x[1])

distance = layers.Lambda(
    compute_distance,
    output_shape=(128,)
)([features_a, features_b])

# Output layer
output = layers.Dense(1, activation="sigmoid")(distance)

# Complete Siamese model
siamese_model = Model(inputs=[input_a, input_b], outputs=output)
siamese_model.compile(
    loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]
)

# Train the model
def train_model():
    train_images_a, train_images_b = train_pairs[:, 0], train_pairs[:, 1]
    test_images_a, test_images_b = test_pairs[:, 0], test_pairs[:, 1]

    # Early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Train the model on the available device (GPU/CPU)
    with tf.device(DEVICE):
        history = siamese_model.fit(
            [train_images_a, train_images_b],
            train_labels,
            validation_data=([test_images_a, test_images_b], test_labels),
            batch_size=32,
            epochs=10,
            callbacks=[early_stopping],
            verbose=1,
        )

    # Save the model
    siamese_model.save("trained_model.h5")
    print("Model saved to 'trained_model.h5'")

    # Evaluate the model
    test_preds = siamese_model.predict([test_images_a, test_images_b])

    # Convert predictions to binary (0 or 1)
    test_preds = (test_preds > 0.5).astype(int)

    # Evaluate metrics
    print("Accuracy:", accuracy_score(test_labels, test_preds))
    print("Precision:", precision_score(test_labels, test_preds))
    print("Recall:", recall_score(test_labels, test_preds))
    print("F1 Score:", f1_score(test_labels, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

# Ensure this is only executed when running the script directly
if __name__ == "__main__":
    train_model()  # This will only run if you run the script directly
