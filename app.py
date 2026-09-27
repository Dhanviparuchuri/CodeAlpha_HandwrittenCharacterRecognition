import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit from 0 to 9.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_digit_model.keras")

def prepare_image(image):
    # Convert the uploaded image to grayscale
    gray = ImageOps.grayscale(image)

    # For dark ink on a light background, invert to match MNIST
    inverted = ImageOps.invert(gray)

    # Remove most of the empty border around the digit
    mask = inverted.point(lambda p: 255 if p > 50 else 0)
    bbox = mask.getbbox()

    if bbox:
        inverted = inverted.crop(bbox)

    # Resize while preserving the digit's proportions
    inverted.thumbnail((20, 20))

    # Center the digit on a black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - inverted.width) // 2
    y = (28 - inverted.height) // 2
    canvas.paste(inverted, (x, y))

    # Normalize and shape the image for the CNN
    pixels = np.array(canvas).astype("float32") / 255.0
    model_input = pixels.reshape(1, 28, 28, 1)

    return canvas, model_input

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, width=300)

    processed_image, model_input = prepare_image(image)

    st.subheader("Image prepared for the model")
    st.image(processed_image, width=200)

    if st.button("Predict digit"):
        model = load_model()
        probabilities = model.predict(model_input, verbose=0)[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        st.success(f"Predicted digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")