import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

tf.config.run_functions_eagerly(True)

st.title("Handwritten Digit Recognition")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("digit_model.keras", compile=False)
    model.make_predict_function()
    return model

model = load_model()

option = st.radio("Choose Input Method", ["Draw Digit", "Upload Image"])


# ---------- DRAW DIGIT ----------
if option == "Draw Digit":

    st.write("Draw a digit below")

    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):

        if canvas_result.image_data is not None:

            img = canvas_result.image_data[:, :, 0]

            img = cv2.resize(img, (28, 28))
            img = img.astype("float32") / 255.0
            img = img.reshape(1, 28, 28, 1)

            prediction = model.predict(img, verbose=0)

            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"Predicted Digit: {digit}")
            st.write(f"Confidence: {confidence:.2f}%")

        else:
            st.warning("Please draw a digit first.")


# ---------- UPLOAD IMAGE ----------
if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload digit image", type=["png","jpg","jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=200)

        img = np.array(image)
        img = 255 - img
        img = cv2.resize(img, (28,28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1,28,28,1)

        prediction = model.predict(img, verbose=0)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}%")