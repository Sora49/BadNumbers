import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
 
# Load model
model = tf.keras.models.load_model("mnist_model.h5")
 
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a digit image (28x28 grayscale) and the model will predict it.")
 
# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = image.resize((28, 28))  # resize
    st.image(image, caption="Uploaded Image", use_column_width=True)
 
    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)
 
    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
 
    st.write(f"### 🔮 Predicted Digit: {predicted_class}")