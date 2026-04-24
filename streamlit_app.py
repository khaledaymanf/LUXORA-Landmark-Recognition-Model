import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="LUXORA AI Recognition", page_icon="🏛️")
st.title("LUXORA: Landmark Recognition")

@st.cache_resource
def load_my_model():
    interpreter = tf.lite.Interpreter(model_path="model/luxora_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_my_model()
class_names = ['Colossoi of Memnon', 'Hatshepsut Temple', 'Karnak_precinct_of_Amun-Ra', 'luxor temple']

uploaded_file = st.file_uploader("Upload an image of a landmark...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
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