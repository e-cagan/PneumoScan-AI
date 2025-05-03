import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="PneumoScan AI", page_icon="🏥", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

st.title("PneumoScan AI: Chest X-Ray Analysis 🏥")
st.markdown("""
<style>
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #FF4B4B, #FFCB05); }
    [data-testid="stFileUploader"] { border: 2px dashed #0068c9; border-radius: 10px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="background:#f0f2f6;padding:15px;border-radius:10px">
        <h4 style="color:#0068c9">🔍 Model Information</h4>
        <a href="https://www.kaggle.com/code/eminaanapaydn/chest-x-ray-images-classification/notebook" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/5968/5968350.png" width="20"> 
            <b>Training Notebook</b>
        </a><br><br>
        <p>Model Accuracy: <b>90%</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ **Medical Disclaimer:**  
    This application is **not for definitive diagnosis**.  
    Results must be evaluated by a medical professional.
    """)

# --- MAIN CONTENT ---
uploaded_file = st.file_uploader("**Upload Chest X-Ray Image** (JPEG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # --- IMAGE PROCESSING ---
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # --- PREDICTION PROCESS ---
        with st.spinner("AI is analyzing..."):
            img = image.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            result = "PNEUMONIA" if confidence > 0.50 else "NORMAL"
            confidence_percent = confidence if result == "PNEUMONIA" else 1 - confidence
            
            # --- RESULTS DISPLAY ---
            with col2:
                st.subheader("Result:")
                if result == "PNEUMONIA":
                    st.error(f"⛔ **PNEUMONIA** ({confidence_percent:.1%} confidence)")
                    st.warning("""
                    **Consult a doctor immediately!**  
                    Findings are consistent with pneumonia.
                    """)
                else:
                    st.success(f"✅ **NORMAL** ({confidence_percent:.1%} confidence)")
                    st.info("""
                    **No need for concern.**  
                    No signs of pneumonia detected in the X-ray.
                    """)
                
                # CONFIDENCE GRAPH (Plotly)
                fig = px.bar(
                    x=["NORMAL", "PNEUMONIA"], 
                    y=[1-confidence, confidence],
                    color=["green", "red"],
                    labels={'x': 'Probability', 'y': 'Percentage'},
                    title="AI Confidence Distribution",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"⛔ Error occurred: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:10px">
    <small>This project was developed for <b>educational purposes</b> only. Not for actual medical diagnosis.</small>
</div>
""", unsafe_allow_html=True)