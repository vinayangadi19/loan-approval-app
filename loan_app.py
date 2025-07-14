import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🏦 Loan Approval Prediction System")
st.write("Predict loan approval status based on applicant details")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    return df

df = load_data()

# Preprocessing
df = df.drop(['Loan_ID'], axis=1)
df.fillna(method='ffill', inplace=True)
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)


label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Applicant Details")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.slider("Applicant Income", 1000, 100000, 5000)
CoapplicantIncome = st.sidebar.slider("Coapplicant Income", 0, 50000, 1000)
LoanAmount = st.sidebar.slider("Loan Amount (in thousands)", 50, 700, 150)
Loan_Amount_Term = st.sidebar.selectbox("Loan Term (in months)", [360, 120, 180, 240, 300, 84, 60, 12])
Credit_History = st.sidebar.selectbox("Credit History", [1, 0])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3])


# Input conversion
user_data = {
    "Gender": 0 if Gender == "Female" else 1,
    "Married": 1 if Married == "Yes" else 0,
    "Education": 0 if Education == "Graduate" else 1,
    "Self_Employed": 1 if Self_Employed == "Yes" else 0,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": 0 if Property_Area == "Rural" else 1 if Property_Area == "Semiurban" else 2,
    "Dependents": Dependents
}


input_df = pd.DataFrame([user_data])
input_df = input_df[X.columns]   # ✅ THIS FIXES THE ORDER ISSUE

# Prediction
if st.button("Check Loan Eligibility"):
    result = model.predict(input_df)[0]
    prediction = "✅ Loan Approved" if result == 1 else "❌ Loan Rejected"
    st.subheader("Prediction Result:")
    st.success(prediction)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: **{acc:.2%}**")
