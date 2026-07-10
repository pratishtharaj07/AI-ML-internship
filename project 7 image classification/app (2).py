import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Binary Image Classifier", page_icon="🖼")

st.title("🖼 Binary Image Classification using CNN")
st.write("Upload an image to classify it.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("binary_image_classifier.keras")

model = load_model()

IMG_SIZE = (150, 150)   # Change if your model was trained on another size

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction")

    # Binary Classification
    if prediction > 0.5:
        st.success(f"Class 1 ({prediction:.2%})")
    else:
        st.success(f"Class 0 ({1-prediction:.2%})")

    st.write(f"Raw Prediction Score: {prediction:.4f}")
