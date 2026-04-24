import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="LUXORA AI Recognition", page_icon="🏛️")
st.title("LUXORA: Landmark Recognition")

@st.cache_resource
def load_tflite_model():
    # تأكد إن المسار صح
    interpreter = tf.lite.Interpreter(model_path="luxora_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error: {e}")
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # results
    score = tf.nn.softmax(predictions[0])
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.success(f"Prediction: **{result}** ({confidence:.2f}%)")
