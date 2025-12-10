# 🔮 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📌 Project Overview
This project aims to predict customer churn in the telecommunications industry using Machine Learning. By analyzing customer demographics, services, and account information, 
the model identifies customers who are likely to leave the company. This helps businesses take proactive measures to retain valuable customers.

## 🚀 Live Demo
Check out the live web application here:
👉 **(https://pkdlskmblsmbmr8acqi9g3.streamlit.app)**

## 📊 Dataset
The project uses the **Telco Customer Churn** dataset from Kaggle.
* **Source:** [Kaggle - BlastChar](https://www.kaggle.com/blastchar/telco-customer-churn)
* **Rows:** 7,043 Customers
* **Target:** `Churn` (Yes/No)

## 🛠️ Technologies Used
* **Python**: Main programming language.
* **Pandas & NumPy**: For data manipulation and analysis.
* **Scikit-learn**: For building the Random Forest model and preprocessing.
* **Streamlit**: For creating the interactive web interface.
* **Seaborn & Matplotlib**: For data visualization.

## 🧠 Model Information
* **Algorithm:** Random Forest Classifier
* **Preprocessing:** Label Encoding for categorical variables.
* **Optimization:** Hyperparameter tuning (n_estimators, max_depth) was applied to prevent overfitting and improve accuracy.
* **Accuracy Score:** **79.53%**

## 📂 Project Structure
```text
├── app.py               # Streamlit application source code
├── churn_model.pkl      # Trained Random Forest model
├── encoders.pkl         # Label encoders for data transformation
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
└── WA_Fn-UseC...csv     # Dataset file

⚙️ How to Run Locally
If you want to run this project on your own machine, follow these steps:

Clone the repository:
git clone [https://github.com/MelikeCirpanli/MIS447-Customer-Churn-Prediction.git](https://github.com/MelikeCirpanli/MIS447-Customer-Churn-Prediction.git)

Install the required libraries:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Developed by: Melike Çırpanlı Student ID: 210106005 Course: MIS 447 Machine Learning
