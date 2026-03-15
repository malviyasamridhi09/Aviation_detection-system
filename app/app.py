
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Aircraft Predictive Maintenance System")
st.caption("AI-powered real-time aircraft health monitoring")
st.divider()
@st.cache_resource
def load_model():
    return joblib.load("models/maintenance_model.pkl")

model = load_model()


col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider("🌡 Temperature", 50, 120, 80)
with col2:
    pressure = st.slider("🔧 Pressure", 90, 120, 105)
with col3:
    vibration = st.slider("📈 Vibration", 0.0, 0.1, 0.05)

st.divider()

if st.button("🚀 Run Maintenance Check"):
    input_data = pd.DataFrame([{
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration
    }])

    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[0][prediction[0]] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ Maintenance Required\n\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Aircraft Operating Normally\n\nConfidence: {confidence:.2f}%")

st.divider()
st.markdown("### 📊 System Highlights")
st.markdown("""
- Machine Learning: **Random Forest**
- Real-time predictions
- Aviation safety focused
- Cloud-deployable
""")
