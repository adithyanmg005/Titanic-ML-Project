import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details:")

# Inputs (must match training columns order)
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
Age = st.slider("Age", 1, 80)
SibSp = st.number_input("Siblings/Spouses", 0, 10)
Parch = st.number_input("Parents/Children", 0, 10)
Fare = st.number_input("Fare", 0.0, 500.0)
Embarked = st.selectbox("Embarked (0 = C, 1 = Q, 2 = S)", [0, 1, 2])

# Prediction
if st.button("Predict"):
    data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    data = scaler.transform(data)
    
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
python -m streamlit run app.py