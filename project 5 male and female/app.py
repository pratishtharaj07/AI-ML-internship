import streamlit as st
import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("FEMALE_MALE_model.pkl")

IMG_SIZE = 64
classes = ["FEMALE", "MALE"]

st.set_page_config(page_title="Gender Prediction", page_icon="👤")

st.title("👤 Gender Prediction using Machine Learning")
st.write("Upload an image to predict whether it is **FEMALE** or **MALE**.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image.")
    else:
        # Preprocess image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()
        img = img.reshape(1, -1)

        # Predict
        prediction = model.predict(img)[0]
        result = classes[prediction]

        # Confidence
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(img)) * 100
            st.success(f"Prediction: **{result}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"Prediction: **{result}**")
