import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the already trained model
model = tf.keras.models.load_model(
    "handwritten_digit_model.keras"
)

# Load the MNIST test dataset
(_, _), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize images
x_test = x_test / 255.0

# Predict the first 10 images together
probabilities = model.predict(
    x_test[:10], verbose=0
)

# Print prediction, actual digit and confidence
for i in range(10):
    predicted = np.argmax(probabilities[i])
    confidence = np.max(probabilities[i]) * 100
    actual = y_test[i]

    print(f"\nImage {i + 1}")
    print(f"Predicted digit: {predicted}")
    print(f"Actual digit: {actual}")
    print(f"Confidence: {confidence:.2f}%")
    print("Correct:", predicted == actual)

# Display the 10 images and their predictions
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    predicted = np.argmax(probabilities[i])

    plt.title(
        f"Pred: {predicted}\nActual: {y_test[i]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()