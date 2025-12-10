import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. SETUP & LOAD ---
st.set_page_config(page_title="Churn Prediction System", page_icon="🔮")

try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'churn_model.pkl' or 'encoders.pkl' not found. Please download them from Kaggle.")
    st.stop()

st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details to predict the risk of churn.")

# --- 2. USER INPUTS ---
st.sidebar.header("Customer Profile")

# Numerical Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 65.0)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])

# Feature Engineering Input (Tenure Group) - Otomatik hesaplanır
def get_tenure_group(m):
    if m <= 12: return 0
    elif m <= 24: return 1
    elif m <= 48: return 2
    elif m <= 60: return 3
    else: return 4

tenure_group = get_tenure_group(tenure)

# Categorical Inputs (Encoder'daki sırayla eşleşmeli)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multilines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# --- 3. PREDICTION LOGIC ---
if st.button("Predict Risk"):
    
    # Senior Citizen dönüşümü (Yes/No -> 1/0)
    senior_val = 1 if senior == "Yes" else 0
    
    # Verileri Encoder kullanarak sayıya çeviriyoruz
    try:
        data = {
            'gender': encoders['gender'].transform([gender])[0],
            'SeniorCitizen': senior_val,
            'Partner': encoders['Partner'].transform([partner])[0],
            'Dependents': encoders['Dependents'].transform([dependents])[0],
            'tenure': tenure,
            'PhoneService': encoders['PhoneService'].transform([phone])[0],
            'MultipleLines': encoders['MultipleLines'].transform([multilines])[0],
            'InternetService': encoders['InternetService'].transform([internet])[0],
            'OnlineSecurity': encoders['OnlineSecurity'].transform([security])[0],
            'OnlineBackup': encoders['OnlineBackup'].transform([backup])[0],
            'DeviceProtection': encoders['DeviceProtection'].transform([device])[0],
            'TechSupport': encoders['TechSupport'].transform([tech])[0],
            'StreamingTV': encoders['StreamingTV'].transform([tv])[0],
            'StreamingMovies': encoders['StreamingMovies'].transform([movies])[0],
            'Contract': encoders['Contract'].transform([contract])[0],
            'PaperlessBilling': encoders['PaperlessBilling'].transform([paperless])[0],
            'PaymentMethod': encoders['PaymentMethod'].transform([payment])[0],
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'tenure_group': tenure_group # Yeni özellik
        }
        
        # DataFrame oluştur
        input_df = pd.DataFrame([data])
        
        # Tahmin yap
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        # Sonuç Göster
        st.divider()
        if prob > 0.5:
            st.error(f"⚠️ HIGH CHURN RISK ({prob*100:.1f}%)")
            st.write("Suggestion: This customer is likely to leave.")
        else:
            st.success(f"✅ LOYAL CUSTOMER ({(1-prob)*100:.1f}%)")
            st.write("Suggestion: This customer is likely to stay.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")