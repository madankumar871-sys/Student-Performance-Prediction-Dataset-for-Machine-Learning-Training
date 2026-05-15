import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Performance Prediction")

st.write("Enter student details below:")

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
previous_score = st.number_input("Previous Score", min_value=0.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0.0)
internet_usage = st.number_input("Internet Usage", min_value=0.0)

if st.button("Predict Exam Score"):

    sample_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "previous_score": [previous_score],
        "assignments_completed": [assignments_completed],
        "internet_usage": [internet_usage]
    })

    sample_scaled = scaler.transform(sample_data)

    prediction = model.predict(sample_scaled)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")