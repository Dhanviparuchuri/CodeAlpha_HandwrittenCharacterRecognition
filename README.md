# Handwritten Character Recognition

## Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9. It is trained on the MNIST dataset and includes a Streamlit app where users can upload an image and get a prediction.

## Features
- Recognizes handwritten digits (0–9)
- Uses a CNN trained on the MNIST dataset
- Displays the predicted digit
- Provides a simple Streamlit interface

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pillow

## Project Files
- `main.py` — trains the model
- `app.py` — runs the Streamlit prediction app
- `demo.py` — demo script
- `evaluate_model.py` — evaluates the model
- `check_outputs.py` — checks generated outputs
- `handwritten_digit_model.keras` — saved trained model

## How to Run

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
