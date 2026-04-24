import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="LUXORA - Landmark Recognition",
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ LUXORA: Landmark Recognition")
st.write("Identify ancient Egyptian landmarks in Luxor using AI.")

@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

MODEL_PATH = "luxora_model.tflite"
interpreter = load_tflite_model(MODEL_PATH)

if interpreter:
    st.success("Model loaded successfully!")

    class_names = ['Colossoi of Memnon', 'Hatshepsut Temple', 'Karnak Precinct', 'Luxor Temple']

    uploaded_file = st.file_uploader("Upload an image of a landmark...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner('Analyzing...'):
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
            probabilities = exp_preds / exp_preds.sum()
            
            predicted_class = class_names[np.argmax(probabilities)]
            confidence = np.max(probabilities) * 100

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.progress(int(confidence))
        st.write(f"Confidence Level: **{confidence:.2f}%**")

else:
    st.warning("Please make sure 'luxora_model.tflite' is in your repository.")

st.markdown("---")
st.caption("Developed by LUXORA Project Team © 2026")
