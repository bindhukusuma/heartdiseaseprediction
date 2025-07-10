
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    return pd.read_csv('data/heart.csv')

data = load_data()

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model/heart_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

st.title("Early Heart Disease Prediction")

def user_input_features():
    age = st.slider('Age', 20, 80, 50)
    sex = st.selectbox('Sex', [0, 1])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.slider('Serum Cholesterol', 100, 400, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Rest ECG (0-2)', [0, 1, 2])
    thalach = st.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.slider('ST depression', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope (0-2)', [0, 1, 2])
    ca = st.selectbox('Major vessels colored (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia (1-3)', [1, 2, 3])
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

input_data = user_input_features()

if st.button('Predict'):
    with open('model/heart_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.subheader("Prediction Result:")
    st.success("🟢 No Heart Disease" if prediction[0] == 0 else "🔴 Heart Disease Detected")
