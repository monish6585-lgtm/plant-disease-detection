import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

with open("class_names.json") as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf photo to detect its disease.")

uploaded_file = st.file_uploader(
    "Upload Diseased Leaf Photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width="stretch")

    if st.button("Predict Disease"):
        img = image.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]
        index = np.argmax(predictions)

        st.success("Predicted Disease: " + class_names[index])
        st.write(f"Confidence: {predictions[index]*100:.2f}%")
