import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("LUXORA: Landmark Recognition")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('luxora_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

class_names = ['Colossoi of Memnon', 'Hatshepsut Temple', 'Karnak Precinct', 'Luxor Temple']

uploaded_file = st.file_uploader("Upload a landmark photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing the landmark...'):
        # preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # predection
        predictions = model.predict(img_array)
        
        # softmax
        score = tf.nn.softmax(predictions[0])
        
        result_label = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

    #result
    st.markdown(f"### Result: **{result_label}**")
    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")
