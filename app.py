import streamlit as st
import pandas as pd
import pickle

# 1. Modeli ve Sütun İsimlerini Yükle
model = pickle.load(open('churn_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# 2. Başlık ve Açıklama
st.title("Müşteri Kayıp Analizi (Churn Prediction)")
st.write("Bu uygulama, bir müşterinin şirketi terk edip etmeyeceğini tahmin eder.")

# 3. Kullanıcıdan Veri Alma (Form)
# Buraya en önemli özellikleri ekledik, diğerleri arka planda 0 olacak.
tenure = st.slider("Müşteri Kaç Aydır Bizimle? (Tenure)", 1, 72, 12)
monthly_charges = st.number_input("Aylık Fatura Tutarı ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Toplam Ödenen Tutar ($)", min_value=0.0, value=500.0)
contract = st.selectbox("Kontrat Tipi", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Ödeme Yöntemi", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 4. Tahmin Butonu
if st.button("Tahmin Et"):
    # Kullanıcının girdiği verileri modelin anlayacağı formata (DataFrame) çeviriyoruz
    # Önce boş bir DataFrame oluştur, sütunları kaydettiğimiz sırayla aynı olsun
    input_data = pd.DataFrame(columns=columns)
    
    # DataFrame'i 0 ile doldur (Varsayılan)
    input_data.loc[0] = 0
    
    # Sayısal Değerleri Yerleştir
    input_data.loc[0, 'tenure'] = tenure
    input_data.loc[0, 'MonthlyCharges'] = monthly_charges
    input_data.loc[0, 'TotalCharges'] = total_charges
    
    # Kategorik (Yazı) Değerleri İşle (One-Hot Encoding Mantığı)
    # Seçilen kutucuğa 1 yazıyoruz.
    
    # Kontrat
    if 'Contract_' + contract in input_data.columns:
        input_data.loc[0, 'Contract_' + contract] = 1
        
    # İnternet
    if 'InternetService_' + internet_service in input_data.columns:
        input_data.loc[0, 'InternetService_' + internet_service] = 1
        
    # Ödeme Yöntemi
    if 'PaymentMethod_' + payment_method in input_data.columns:
        input_data.loc[0, 'PaymentMethod_' + payment_method] = 1

    # TAHMİN YAP
    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1] # Bu, müşterinin GİTME ihtimali

    # Sonucu Göster
    st.divider()
    
    if prediction == 1:
        # Eğer tahmin "Gidecek" ise, zaten gitme ihtimalini (örn: %80) yazdırıyoruz. Doğru.
        st.error(f"⚠️ DİKKAT: Bu müşteri %{churn_probability*100:.1f} ihtimalle TERK EDECEK!")
    else:
        # DÜZELTME BURADA:
        # Eğer tahmin "Kalacak" ise, Gitme ihtimali (örn: %26) düşüktür.
        # Ekrana KALMA ihtimalini yazdırmak için (100 - 26 = %74) işlemini yapmalıyız.
        stay_probability = 1 - churn_probability
        st.success(f"✅ GÜVENLİ: Bu müşteri %{stay_probability*100:.1f} ihtimalle KALACAK.")