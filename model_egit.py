import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Veriyi Oku
print("Veri okunuyor...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Veriyi Temizle (Aynı işlemleri burada da yapıyoruz)
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df)

# 3. Modeli Hazırla
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Eğit
print("Model eğitiliyor...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Kaydet (Artık senin bilgisayarının sürümüyle kaydedilecek!)
pickle.dump(model, open('churn_model.pkl', 'wb'))
pickle.dump(X.columns, open('columns.pkl', 'wb'))

print("Tebrikler! Model ve sütunlar başarıyla kaydedildi.")