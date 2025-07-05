import streamlit as st
import numpy as np
import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="🩺 Pro Health Risk Predictor", page_icon="💊", layout="centered")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #f9fbfc;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #1976D2;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        .stProgress > div > div > div > div {
            background-color: #FF6F61;
        }
        hr {
            border-top: 1px solid #bbb;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("""
This app predicts diabetes risk using **machine learning**.

- Based on **PIMA Diabetes dataset**
- By **Siva** ❤️
""")
st.sidebar.markdown("## 🔒 Privacy")
st.sidebar.caption("This app **does not store** any personal data. All inputs stay local.")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💊 Pro Health Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("Check your diabetes risk with personalized insights and health tips.")
st.markdown("---")

with st.form("health_form"):
    st.markdown("### 🔍 Enter Your Health Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        preg = st.number_input("👶 Pregnancies", 0, 20, 1)
        skin = st.number_input("🧪 Skin Thickness", 0, 100, 20)
        dpf = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    with col2:
        gluc = st.number_input("🍬 Glucose", 0, 200, 100)
        insulin = st.number_input("💉 Insulin", 0, 900, 80)
        age = st.number_input("🎂 Age", 1, 120, 30)
    with col3:
        bp = st.number_input("💓 Blood Pressure", 0, 150, 70)
        bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0, step=0.1)
    submitted = st.form_submit_button("🚀 Predict Now")

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "🔵"
    elif 18.5 <= bmi < 25: return "Normal", "🟢"
    elif 25 <= bmi < 30: return "Overweight", "🟠"
    else: return "Obese", "🔴"

if submitted:
    st.markdown("---")
    input_data = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction] * 100
    bmi_category, bmi_icon = get_bmi_category(bmi)

    st.subheader("🧾 Your Health Summary")
    st.success(f"""
    - **Pregnancies**: {preg}
    - **Glucose**: {gluc}
    - **Blood Pressure**: {bp}
    - **Skin Thickness**: {skin}
    - **Insulin**: {insulin}
    - **BMI**: {bmi} → {bmi_icon} *{bmi_category}*
    - **Diabetes Pedigree Function**: {dpf}
    - **Age**: {age}
    """)

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes ({probability:.2f}%)")
    else:
        st.success(f"✅ Low risk of diabetes ({probability:.2f}%)")

    st.markdown("### 🧠 Personalized Suggestions")
    if prediction == 1:
        st.warning("🔹 Consult a physician\n🔹 Reduce sugar\n🔹 Walk 30 mins daily\n🔹 Regular checkups suggested")
    else:
        st.info("✅ Keep up your routine!\n✅ Annual checkup advised\n✅ Balanced diet & sleep")

    st.markdown("### 🧭 Health Risk Meter")
    st.progress(int(probability))

    st.markdown("### 📈 Your Key Metrics")
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ['Glucose', 'BP', 'BMI', 'Age']
    values = [gluc, bp, bmi, age]
    ax.bar(labels, values, color=['#FF8A80', '#4FC3F7', '#81C784', '#BA68C8'])
    ax.set_ylabel("Value")
    ax.set_title("Health Parameters")
    st.pyplot(fig)

    st.markdown("### 📋 Risk Interpretation")
    risk_data = {
        "Parameter": ["Glucose", "BP", "BMI", "Age"],
        "Your Value": [gluc, bp, bmi, age],
        "Healthy Range": ["70–140", "80–120", "18.5–24.9", "N/A"],
        "Remarks": [
            "✅ Normal" if 70 <= gluc <= 140 else "⚠️ Abnormal",
            "✅ Normal" if 80 <= bp <= 120 else "⚠️ Abnormal",
            "✅ Normal" if 18.5 <= bmi <= 24.9 else "⚠️ Needs Attention",
            "🎂 Age-based risk"
        ]
    }
    st.dataframe(pd.DataFrame(risk_data))

    if 40 <= probability <= 60:
        st.warning("⚠️ Medium confidence prediction. Consider professional evaluation.")

    st.markdown("### 📆 Personalized Health Check Reminder")

    if prediction == 1:
        if age > 50 or bmi > 30:
            days_until_next = 30
        elif age > 35:
            days_until_next = 90
        else:
            days_until_next = 180
    else:
        days_until_next = 180

    final_days = min(days_until_next + 15, 180)
    next_check = datetime.today() + timedelta(days=final_days)
    st.info(f"📅 **Next recommended checkup:** `{next_check.strftime('%B %d, %Y')}`")

    df = pd.DataFrame({
        "Timestamp": [datetime.now()],
        "Preg": [preg], "Glucose": [gluc], "BP": [bp], "Skin": [skin],
        "Insulin": [insulin], "BMI": [bmi], "DPF": [dpf], "Age": [age],
        "Prediction": ["High" if prediction == 1 else "Low"],
        "Probability": [round(probability, 2)]
    })

    try:
        existing = pd.read_csv("history.csv")
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv("history.csv", index=False)
    st.download_button("📥 Download Report", df.to_csv(index=False), file_name="health_report.csv")

st.markdown("<hr><center style='color:gray;'>Made with ❤️ by <b>Siva</b> | Powered by Scikit-learn + Streamlit</center>", unsafe_allow_html=True)
