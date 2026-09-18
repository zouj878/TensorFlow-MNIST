import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib to display Chinese characters and fix minus sign rendering warning
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese text
plt.rcParams['axes.unicode_minus'] = False    # Fix negative symbol display issue

# ===================== 1. Load and preprocess MNIST dataset =====================
# Automatically download dataset (about 10MB for the first run; load locally afterwards)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data preprocessing: normalize pixel values from range 0-255 to 0-1 to improve training stability
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print basic dataset information for inspection
print("✅ Dataset loaded successfully:")
print(f"Training set: {x_train.shape} (60000 handwritten digit images of size 28×28)")
print(f"Test set: {x_test.shape} (10000 held-out test images)")

# ===================== 2. Build simple neural network model =====================
model = tf.keras.Sequential([
    # Flatten layer: convert 2D 28×28 image into a 1D vector of length 784 (input layer)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Hidden layer: 128 neurons with ReLU activation function to introduce non-linearity
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer: 10 neurons corresponding to digits 0-9, softmax outputs class probabilities
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model: define optimizer, loss function and evaluation metric
model.compile(
    optimizer='adam',  # Popular adaptive optimizer with auto learning rate adjustment
    loss='sparse_categorical_crossentropy',  # Loss function suitable for integer-label classification
    metrics=['accuracy']  # Use classification accuracy as evaluation metric
)

# ===================== 3. Train the model =====================
print("\n🚀 Starting model training (5 epochs for quick demonstration):")
history = model.fit(
    x_train, y_train,
    epochs=5,  # Training epochs; higher epochs usually improve accuracy. 5 balances speed and performance
    validation_split=0.1  # Reserve 10% of training data as validation set to monitor overfitting
)

# ===================== 4. Evaluate model performance =====================
print("\n📊 Model evaluation on test set:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test set accuracy: {test_acc:.4f} (typically above 97%)")

# ===================== 5. Visualization: randomly select 5 test images and run prediction =====================
# Remove fixed random seed so sampled images differ each run
np.random.seed(None)
# Randomly sample 5 unique indices from test set
random_idx = np.random.choice(len(x_test), 5, replace=False)

# Generate predictions for the 5 selected images
predictions = model.predict(x_test[random_idx])
# Extract class with maximum probability as predicted label
predicted_labels = np.argmax(predictions, axis=1)

# Plot images with ground truth labels and predicted labels
plt.figure(figsize=(10, 4))
for i, idx in enumerate(random_idx):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')  # Display selected handwritten digit image
    plt.title(f"True: {y_test[idx]}\nPred: {predicted_labels[i]}")
    plt.axis('off')  # Hide axis ticks and borders
plt.tight_layout()
plt.show()

print("\n🎉 The model can accurately recognize various handwritten digits.")
