import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

def load_model(model_path):
    return keras.models.load_model(model_path)

def preprocess_image(image):
    img = image.resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict(model, img_array):
    return model.predict(img_array)

def interpret_prediction(prediction, class_names):
    predicted_class_index = np.argmax(prediction)
    return class_names[predicted_class_index], prediction[0][predicted_class_index]

def display_result(col, model_name, confidence, predicted_class, base_confidence=None):
    confidence_str = f'{confidence * 100:.2f}%'
    col.markdown(f'<h5>{model_name}</h5>', unsafe_allow_html=True)
    col.markdown(f'<p>Prediksi: {predicted_class}</p>', unsafe_allow_html=True)
    
    if base_confidence is None:
        col.metric(label="Tingkat Keyakinan", value=confidence_str)
    else:
        gap_confidence = (confidence - base_confidence) * 100
        delta_str = f"{gap_confidence:.2f}% {'lebih tinggi' if gap_confidence > 0 else 'lebih rendah'}"
        col.metric(label="Tingkat Keyakinan", value=confidence_str, delta=delta_str)

st.markdown("<h1 style='text-align: center;'>Identifikasi Penyakit Daun Tanaman Jagung</h1>", unsafe_allow_html=True)

class_names = ['Blight', 'Common Rust', 'Grey Leaf Spot', 'Healthy']
models = {
    'Tanpa Augmentasi Data': load_model('no_aug.keras'),
    'Dengan Augmentasi Geometri': load_model('geo_aug.keras'),
    'Dengan Augmentasi Fotometri': load_model('photo_aug.keras')
}

uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    processed_image = preprocess_image(image)
    
    results = {name: interpret_prediction(predict(model, processed_image), class_names) 
               for name, model in models.items()}
    
    cols = st.columns(3)
    base_confidence = results['Tanpa Augmentasi Data'][1]
    
    for (name, (pred_class, conf)), col in zip(results.items(), cols):
        display_result(col, f"Model {name}", conf, pred_class, base_confidence if name != 'Tanpa Augmentasi Data' else None)