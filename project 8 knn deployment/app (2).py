import streamlit as st
import joblib
import numpy as np
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_knn.pkl")

model = joblib.load(MODEL_PATH)

st.title("🪷 Iris Flower Classifier")

st.write("Enter flower measurements")

sepal_length = st.number_input("Sepal Length (cm)", 0.0,10.0,5.1)
sepal_width = st.number_input("Sepal Width (cm)",0.0,10.0,3.5)
petal_length = st.number_input("Petal Length (cm)",0.0,10.0,1.4)
petal_width = st.number_input("Petal Width (cm)",0.0,10.0,0.2)

if st.button("Predict"):

    sample = np.array([[sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width]])

    prediction = model.predict(sample)[0]

    species = {
        0:"🌸 Setosa",
        1:"🌺 Versicolor",
        2:"🪷Virginica"
    }

    st.success(f"Prediction : {species[prediction]}")
