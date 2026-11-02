import streamlit as st
import tensorflow as tf
import numpy as np
import json
import csv
import io
import os
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="PlantCare AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------
# Model and class-name loading
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Keep it in the same GitHub repository folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Could not find {CLASS_NAMES_PATH}. Keep it in the same GitHub repository folder as app.py."
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    # Supports either a list or a JSON dictionary of index -> class name.
    if isinstance(names, dict):
        try:
            names = [names[str(i)] for i in range(len(names))]
        except (KeyError, TypeError):
            names = list(names.values())
    return names

try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(f"Unable to load the model or class names: {exc}")
    st.info(
        "Check that app.py, plant_disease_model.keras and class_names.json "
        "are in the same repository directory, and that requirements.txt is installed."
    )
    st.stop()

# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------
def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)

def predict_image(image):
    probabilities = model.predict(prepare_image(image), verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float32).flatten()

    # Handle models that return logits rather than probabilities.
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(float(np.sum(probabilities)), 1.0, atol=0.05)
    ):
        probabilities = tf.nn.softmax(probabilities).numpy()

    count = min(len(class_names), len(probabilities))
    order = np.argsort(probabilities[:count])[::-1]
    return [(class_names[int(i)], float(probabilities[int(i)])) for i in order]

def estimate_severity(image):
    """
    A simple visual heuristic based on yellow/brown pixels.
    It is an approximate indicator only, not a validated agronomic diagnosis.
    """
    rgb = np.asarray(image.convert("RGB").resize((256, 256)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    yellow = cv2.inRange(hsv, np.array([15, 35, 35]), np.array([40, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 35, 20]), np.array([25, 255, 180]))
    affected = cv2.bitwise_or(yellow, brown)

    # Ignore very dark pixels, which are often background/shadows.
    valid = cv2.inRange(hsv, np.array([0, 20, 25]), np.array([179, 255, 255]))
    valid_pixels = max(int(np.count_nonzero(valid)), 1)
    affected_pixels = int(np.count_nonzero(cv2.bitwise_and(affected, valid)))
    ratio = min(affected_pixels / valid_pixels, 1.0)

    if ratio < 0.08:
        label = "Low"
    elif ratio < 0.22:
        label = "Moderate"
    else:
        label = "High"
    return label, ratio * 100

def make_gradcam_overlay(image):
    """
    Attempts Grad-CAM for common CNN / MobileNetV2 model structures.
    Returns None if a suitable convolutional layer cannot be located.
    """
    try:
        target_layer = None
        target_model = model
        nested_base = None

        # Search top-level layers, then common nested feature extractors.
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break
            if isinstance(layer, tf.keras.Model):
                for inner in reversed(layer.layers):
                    if isinstance(inner, tf.keras.layers.Conv2D):
                        target_layer = inner
                        nested_base = layer
                        break
                if target_layer is not None:
                    break

        if target_layer is None:
            return None

        if nested_base is not None:
            grad_model = tf.keras.Model(
                inputs=nested_base.input,
                outputs=[target_layer.output, nested_base.output],
            )
            # This path works when the nested feature extractor is directly
            # connected to the model input. Otherwise, skip safely.
            if len(model.inputs) != 1:
                return None
            x = tf.cast(prepare_image(image), tf.float32)
            with tf.GradientTape() as tape:
                conv_outputs, features = grad_model(x, training=False)
                head = features
                found = False
                for layer in model.layers:
                    if layer is nested_base:
                        found = True
                        continue
                    if found:
                        head = layer(head, training=False)
                score = head[:, tf.argmax(head[0])]
        else:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
            x = tf.cast(prepare_image(image), tf.float32)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x, training=False)
                class_index = tf.argmax(predictions[0])
                score = predictions[:, class_index]

        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap[0], 0)
        max_value = tf.reduce_max(heatmap)
        if float(max_value) == 0:
            return None
        heatmap = heatmap / max_value
        heatmap = cv2.resize(heatmap.numpy(), image.size)
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        original = np.asarray(image.convert("RGB"))
        overlay = np.uint8(0.55 * original + 0.45 * colored)
        return Image.fromarray(overlay)
    except Exception:
        return None

def disease_guidance(disease_name):
    name = disease_name.lower()
    if "healthy" in name:
        return (
            "The model classified this leaf as healthy. Continue regular crop monitoring, "
            "appropriate watering, and good field hygiene."
        )
    if any(word in name for word in ["blight", "rot"]):
        return (
            "Remove heavily affected plant material where appropriate, avoid overhead watering, "
            "improve airflow, and confirm the diagnosis with a local agricultural expert before treatment."
        )
    if any(word in name for word in ["rust", "mildew", "mold"]):
        return (
            "Monitor nearby leaves, reduce prolonged leaf wetness, improve spacing and airflow, "
            "and seek locally approved treatment advice after confirming the disease."
        )
    if any(word in name for word in ["spot", "scab"]):
        return (
            "Avoid handling wet plants, remove fallen infected debris where appropriate, "
            "and monitor whether spots spread. Confirm the diagnosis before applying treatment."
        )
    return (
        "Inspect other leaves and the whole plant, maintain suitable watering and airflow, "
        "and confirm the model result with an agricultural expert before taking treatment decisions."
    )

# ---------------------------------------------------------
# Sidebar and page header
# ---------------------------------------------------------
with st.sidebar:
    st.title("🌱 PlantCare AI")
    st.caption("Deep-learning-based plant disease screening")
    page = st.radio(
        "Navigate",
        ["Home & Detection", "Batch Analysis", "About the Project"],
    )
    st.divider()
    st.caption("Model: MobileNetV2")
    st.caption("Input size: 224 × 224")
    st.caption("Dataset classes: " + str(len(class_names)))

st.title("🌿 AI-Based Plant Disease Detection")
st.write(
    "Upload a plant leaf image to obtain a model prediction, confidence scores, "
    "an approximate visual severity indicator and an optional Grad-CAM explanation."
)
st.warning(
    "Research prototype: predictions and severity estimates are not a substitute "
    "for diagnosis by an agricultural expert. Performance on clean dataset images "
    "may not represent performance in real field conditions."
)

# ---------------------------------------------------------
# Home & Detection
# ---------------------------------------------------------
if page == "Home & Detection":
    left, right = st.columns([1, 1])
    with left:
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded leaf image", use_container_width=True)

    with right:
        if uploaded_file:
            if st.button("🔍 Analyse Leaf", type="primary", use_container_width=True):
                with st.spinner("Analysing image..."):
                    results = predict_image(image)
                    best_name, best_score = results[0]
                    severity, severity_percent = estimate_severity(image)

                st.success(f"Predicted class: {best_name}")
                metric1, metric2 = st.columns(2)
                metric1.metric("Top prediction confidence", f"{best_score * 100:.2f}%")
                metric2.metric("Visual severity estimate", severity)

                st.subheader("Top-5 predictions")
                top_df = pd.DataFrame(
                    [{"Disease / Class": n, "Confidence (%)": round(p * 100, 2)}
                     for n, p in results[:5]]
                )
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                st.subheader("Visual severity estimate")
                st.progress(min(int(round(severity_percent)), 100))
                st.caption(
                    f"Approximate yellow/brown pixel proportion: {severity_percent:.1f}% "
                    "(heuristic estimate, not validated disease severity)."
                )

                st.subheader("Suggested next steps")
                st.write(disease_guidance(best_name))

                with st.expander("Show Grad-CAM explanation"):
                    overlay = make_gradcam_overlay(image)
                    if overlay is not None:
                        st.image(
                            overlay,
                            caption="Grad-CAM heatmap overlay (model attention explanation)",
                            use_container_width=True,
                        )
                        st.caption(
                            "The heatmap highlights image regions influencing the model. "
                            "It is not a lesion segmentation mask."
                        )
                    else:
                        st.info(
                            "Grad-CAM could not be generated for this model structure. "
                            "The prediction is still available."
                        )

                report = pd.DataFrame([{
                    "filename": uploaded_file.name,
                    "predicted_class": best_name,
                    "confidence_percent": round(best_score * 100, 2),
                    "severity_estimate": severity,
                    "yellow_brown_pixel_percent": round(severity_percent, 2),
                }])
                st.download_button(
                    "⬇️ Download prediction report (CSV)",
                    data=report.to_csv(index=False).encode("utf-8"),
                    file_name="prediction_report.csv",
                    mime="text/csv",
                )
        else:
            st.info("Upload a JPG, JPEG or PNG image to begin.")

# ---------------------------------------------------------
# Batch Analysis
# ---------------------------------------------------------
elif page == "Batch Analysis":
    st.subheader("Analyse multiple leaf images")
    files = st.file_uploader(
        "Upload multiple leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if files and st.button("Run Batch Analysis", type="primary"):
        rows = []
        progress = st.progress(0)
        for i, file in enumerate(files):
            try:
                image = Image.open(file).convert("RGB")
                results = predict_image(image)
                severity, severity_percent = estimate_severity(image)
                rows.append({
                    "filename": file.name,
                    "predicted_class": results[0][0],
                    "confidence_percent": round(results[0][1] * 100, 2),
                    "severity_estimate": severity,
                    "yellow_brown_pixel_percent": round(severity_percent, 2),
                })
            except Exception as exc:
                rows.append({
                    "filename": file.name,
                    "predicted_class": "Processing error",
                    "confidence_percent": None,
                    "severity_estimate": "Unavailable",
                    "yellow_brown_pixel_percent": None,
                    "error": str(exc),
                })
            progress.progress((i + 1) / len(files))

        batch_df = pd.DataFrame(rows)
        st.subheader("Batch prediction report")
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download batch report (CSV)",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_prediction_report.csv",
            mime="text/csv",
        )

        valid = batch_df[batch_df["predicted_class"] != "Processing error"]
        if not valid.empty:
            st.subheader("Predicted class distribution")
            counts = valid["predicted_class"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Number of images")
            ax.set_title("Batch prediction summary")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------
# About
# ---------------------------------------------------------
else:
    st.subheader("About this project")
    st.write(
        "This project uses transfer learning with MobileNetV2 to classify plant leaf "
        "images into the classes represented in the PlantVillage dataset."
    )
    st.markdown(
        """
        **Main features**
        - Plant leaf image upload and disease-class prediction
        - Top-5 prediction confidence scores
        - Approximate colour-based severity indicator
        - Grad-CAM visual explanation when supported by the model structure
        - Batch image analysis and downloadable CSV report

        **Technology stack**
        - Python and Streamlit
        - TensorFlow / Keras
        - MobileNetV2
        - NumPy, Pandas, Pillow and OpenCV
        - Matplotlib

        **Important limitations**
        - Dataset validation accuracy is not the same as real-world field accuracy.
        - The colour-based severity estimate is a simple heuristic, not a clinically or agronomically validated measure.
        - Grad-CAM is an explanation of model activation, not a precise map of infected tissue.
        - Treatment guidance is general and should be checked with a local agricultural expert.
        """
    )
    st.info(
        f"The app loaded {len(class_names)} class labels. "
        "The model and class_names.json should be stored beside app.py in the deployment repository."
    )

