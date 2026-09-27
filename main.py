import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Normalize pixel values (0-255 to 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 6. Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=2
)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 7. Save the trained model
model.save("handwritten_digit_model.keras")

# 8. Check predictions for 10 test images
for i in range(10):
    image = x_test[i:i + 1]
    probabilities = model.predict(image, verbose=0)

    predicted_digit = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100
    actual_digit = y_test[i]

    print(f"\nImage {i + 1}")
    print(f"Predicted: {predicted_digit}")
    print(f"Actual: {actual_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print("Correct:", predicted_digit == actual_digit)

# 9. Display the test image
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()