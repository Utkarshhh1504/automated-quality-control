import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Mobile.h5")

# Same order the notebook used when training
CLASS_NAMES = ["ground_truth_1", "ground_truth_2", "oil", "scratch", "stain", "good"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


def predict_image(pil_img):
    # Match the training pipeline: grayscale -> 32x32 -> 192x108 -> 3 channels, no /255
    gray = np.array(pil_img.convert("L"))
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.resize(gray, (192, 108))
    img = np.repeat(img[..., np.newaxis], 3, axis=-1).astype("float32")
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = int(np.argmax(preds, axis=1)[0])
    return CLASS_NAMES[idx], float(preds[0][idx])


st.title("Mobile Screen Damage Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    label, confidence = predict_image(img)
    st.write(f"Prediction: **{label}** ({confidence:.1%} confidence)")
