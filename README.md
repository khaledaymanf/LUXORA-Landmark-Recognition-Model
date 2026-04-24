# 🏛️ LUXORA: AI-Powered Egyptian Landmark Recognition
**LUXORA** is a deep learning-based project designed to identify and classify prominent ancient Egyptian landmarks in Luxor. Using state-of-the-art Computer Vision techniques, the project provides a seamless bridge between cultural heritage and modern technology, deployed via a Flutter mobile application and a Streamlit web interface.

---

## 📸 Key Features
* **High Accuracy Recognition:** Powered by the **ConvNeXt** architecture.
* **Real-time Inference:** Optimized for mobile using **TFLite**.
* **User-Friendly Interfaces:** Available on both Mobile (Flutter) and Web (Streamlit).

---

## 📊 Dataset & Processing
The model is trained to recognize 4 primary classes:
1.  **Colossoi of Memnon**
2.  **Hatshepsut Temple**
3.  **Karnak Precinct of Amun-Ra**
4.  **Luxor Temple**

### Data Pipeline:
* **Splitting:** 80% Training | 20% Validation.
* **Preprocessing:** Images resized to `224x224`, normalized to `[0, 1]`, and Center-Cropped to focus on architectural features.
* **Optimization:** Used `tf.data.AUTOTUNE`, caching, and prefetching for high-performance training.

---

## 🧠 Technical Architecture
* **Base Model:** ConvNeXt (Transfer Learning from ImageNet-1k).
* **Technique:** Fine-tuning top layers to adapt to archaeological textures and structures.
* **Evaluation Metrics:**
    * **Validation Accuracy:** ~84%
    * **Loss:** 0.47 (Stable convergence)
    * **Top Performer:** Hatshepsut Temple (F1-Score: 0.96)

---

## 🚀 Deployment

### Mobile App 
### Web App (Streamlit) -> 
