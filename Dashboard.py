import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="🩺 Smartwatch Health Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Predict Health", "ℹ️ About"])

# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load("smartwatch_heart_model.pkl")
    df = pd.read_csv("heart_disease_data.csv")
    return model, df

model, df = load_resources()

# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.title("💡 Smartwatch Health Monitoring Dashboard")
    st.markdown("<h4 style='color:black;'>🌐 AI-Powered Cardiovascular Insights from Your Wrist</h4>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:black;'>Track, analyze, and predict heart health using wearable-style data features.</span>", unsafe_allow_html=True)

    with st.expander("📊 Preview Dataset"):
        st.dataframe(df.head(50))

# ---------------- Prediction Page ----------------
elif page == "🧠 Predict Health":
    st.title("🧠 Predict Your Heart Health")
    st.markdown("<h4 style='color:black;'>📥 Enter Your Smartwatch Health Metrics Below</h4>", unsafe_allow_html=True)

    with st.form("health_form"):
        col1, col2 = st.columns(2)

        with col1:
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 40, 180, 75)
            spo2 = st.slider("🫁 Oxygen Saturation (SpO2 %)", 85.0, 100.0, 97.0)
            systolic = st.slider("🔴 Systolic BP (mm Hg)", 90, 180, 120)
            diastolic = st.slider("🔵 Diastolic BP (mm Hg)", 60, 120, 80)
            sleep = st.slider("😴 Sleep Duration (hrs)", 0.0, 12.0, 7.0)
            stress = st.slider("😰 Stress Level (1–10)", 1, 10, 5)

        with col2:
            steps = st.number_input("👟 Steps Count (today)", 0, 30000, 5000)
            calories = st.number_input("🔥 Calories Burned", 1000, 5000, 2500)
            hrv = st.slider("📉 Heart Rate Variability (ms)", 10.0, 120.0, 60.0)
            skin_temp = st.slider("🌡️ Skin Temperature (°C)", 34.0, 38.0, 36.5)
            resp_rate = st.slider("🌬️ Respiration Rate (bpm)", 10, 30, 16)
            vo2 = st.slider("💪 VO2 Max", 20.0, 60.0, 40.0)

        submitted = st.form_submit_button("🧠 Predict Health Status")

        if submitted:
            input_df = pd.DataFrame([{
                "Heart_Rate_bpm": heart_rate,
                "SpO2_%": spo2,
                "Blood_Pressure_Systolic": systolic,
                "Blood_Pressure_Diastolic": diastolic,
                "Sleep_Duration_hr": sleep,
                "Steps_Count": steps,
                "Calories_Burned": calories,
                "Stress_Level": stress,
                "HRV_ms": hrv,
                "Skin_Temperature_C": skin_temp,
                "Respiration_Rate_bpm": resp_rate,
                "VO2_Max": vo2
            }])

            prediction = model.predict(input_df)[0]
            result_text = "🟢 Healthy Heart – No Immediate Risk" if prediction == 0 else "🔴 At Risk – Consult a Healthcare Professional"

            st.markdown("### ✅ Prediction Result")
            if prediction == 1:
                st.error(result_text)
            else:
                st.success(result_text)

# ---------------- About Page ----------------
elif page == "ℹ️ About":
    st.title("📘 About This Project")
    st.markdown("""
    <span style='color:black'><b>📌 Project Title:</b></span> Smartwatch-Based Heart Health Monitoring using AI  
    <span style='color:black'><b>🔧 Built With:</b></span> Python, Streamlit, Scikit-learn, and TensorFlow  
    <span style='color:black'><b>🎯 Goal:</b></span> Real-time prediction of heart health using smartwatch-style data

    <br>
    <span style='color:black'><b>💡 Key Features:</b></span>
    - Uses common smartwatch metrics (HR, SpO2, BP, Sleep, etc.)  
    - Predicts cardiovascular risk with machine learning  
    - Lightweight, responsive, and intuitive UI
    """, unsafe_allow_html=True)

# ---------------- Custom Styling ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        font-family: 'Segoe UI', sans-serif;
        color: black;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .stForm {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stSlider label, .stNumberInput label {
        color: black;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
