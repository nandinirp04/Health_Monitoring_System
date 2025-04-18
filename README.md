# Health_Monitoring_System
# 🩺 Smartwatch-Based Heart Health Monitoring System

An AI-powered health monitoring system that uses smartwatch-style data to predict heart health risks. Built with a custom Streamlit dashboard and a machine learning model trained on synthetically generated health metrics.

---

## 🚀 Features

- 🔍 **Real-time heart disease risk prediction**
- 📊 **Interactive Streamlit dashboard UI**
- 📱 **Smartwatch-compatible features** like:
  - Heart Rate
  - Blood Pressure
  - SpO2
  - Sleep Duration
  - HRV
  - Stress Level
  - VO2 Max
- 🤖 Machine Learning model trained using **TensorFlow**
- 🎨 Professionally styled dashboard with light UI/UX

---

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- TensorFlow
- Joblib
- Matplotlib / Seaborn (optional for charts)
- `streamlit-option-menu` for sidebar navigation

---

## 📁 Project Structure

```bash
.
├── Dashboard.py                 # Streamlit dashboard UI
├── train_model.py              # Model training script
├── smartwatch_heart_model.pkl  # Trained ML model
├── heart_disease_data.csv      # Synthetic dataset
├── Heart_Disease.csv           # Additional formatted dataset (if used)
├── model.json.ipynb            # Notebook with model structure or architecture
├── README.md                   # You're here
