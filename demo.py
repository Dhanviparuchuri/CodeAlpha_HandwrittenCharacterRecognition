import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("handwritten_digit_model.keras")

# Load test images and labels
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images
x_test = x_test / 255.0

# Show predictions for 9 test images
plt.figure(figsize=(8, 8))

for i in range(9):
    image = x_test[i]
    prediction = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
    predicted_digit = np.argmax(prediction)

    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted: {predicted_digit} | Actual: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()