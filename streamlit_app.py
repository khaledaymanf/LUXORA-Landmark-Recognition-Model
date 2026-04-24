import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.special import softmax

st.set_page_config(page_title="LUXORA AI", page_icon="🏛️")
st.title("🏛️ LUXORA: Landmark Recognition")

@st.cache_resource
def load_tflite_model():
    model_path = "luxora_model.tflite"
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

interpreter = load_tflite_model()

class_names = [
    'Colossoi of Memnon', 
    'Hatshepsut Temple', 
    'Karnak Precinct', 
    'Luxor Temple'
]

# upload image
uploaded_file = st.file_uploader("Upload an Egyptian Landmark photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    # image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Calculating...'):
        # Preprocessing
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Resize 
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)

        # preprocessing
        img_array = tf.keras.applications.convnext.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        #Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        raw_predictions = interpreter.get_tensor(output_details[0]['index'])
        
        probabilities = softmax(raw_predictions[0])
        
        result_index = np.argmax(probabilities)
        confidence = probabilities[result_index] * 100

    #result
    if confidence > 50: 
        st.success(f"### Landmark: **{class_names[result_index]}**")
        st.write(f"**Confidence Level:** {confidence:.2f}%")
        st.progress(int(confidence))
    else:
        st.warning("⚠️ Could not recognize the landmark with high confidence. Please try a closer/clearer shot.")

# تذييل
st.markdown("---")
st.info("Tip: Make sure the image is well-lit and focused on the main structure.")
