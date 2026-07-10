import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Male/Female Eye Classifier",
    page_icon="👁",
    layout="centered"
)

st.title("👁 Male/Female Eye Classification")
st.write("Upload an eye image to predict whether it belongs to a Male or Female.")

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose an Eye Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    st.subheader("Prediction")

    # Change this if your class order is different
    if prediction >= 0.5:
        st.success("👨 Male Eye")
        st.write(f"Confidence: {prediction*100:.2f}%")
    else:
        st.success("👩 Female Eye")
        st.write(f"Confidence: {(1-prediction)*100:.2f}%")

    st.progress(float(max(prediction, 1-prediction)))
