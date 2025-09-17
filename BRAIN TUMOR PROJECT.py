import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("brain_tumor_model.keras")
    return model

model = load_trained_model()

# Classes (update if you have more tumor types)
classes = ['brain_tumor', 'good_health']  
tumor_types = ['glioma', 'meningioma', 'pituitary']  # if you have multi-class tumor labels

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect if a brain tumor is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # Predict
    pred = model.predict(x)
    pred_class = classes[np.argmax(pred[0])]
    pred_confidence = pred[0][np.argmax(pred[0])] * 100
    
    st.write(f"**Prediction:** {pred_class} ({pred_confidence:.2f}%)")
    
    # If tumor, ask user for tumor type (or predict if your model supports multi-class)
    if pred_class == 'brain_tumor':
        st.write("Detected as a tumor.")
        # Optional: if your model can detect tumor type
        # pred_tumor_type = tumor_types[np.argmax(pred[0][1:])]  # adjust indexing if multi-class
        # st.write(f"Tumor Type: {pred_tumor_type}")
