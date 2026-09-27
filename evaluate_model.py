import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model("handwritten_digit_model.keras")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

# Add channel dimension for the CNN
x_test = x_test[..., np.newaxis]

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

probabilities = model.predict(x_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Create and save the confusion matrix image
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

fig, ax = plt.subplots(figsize=(10, 8))
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("MNIST Digit Recognition - Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
print("Confusion matrix saved as confusion_matrix.png")
plt.show()