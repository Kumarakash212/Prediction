import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and scaler
scaler = joblib.load('scaler.pkl')
dropout_model = joblib.load('dropout_model.pkl')
stress_model = joblib.load('stress_model.pkl')

st.title("Student Dropout & Mental Stress Predictor")

# Input form
grades = st.slider("Grades (0-100)", 0, 100, 70)
attendance = st.slider("Attendance (%)", 0, 100, 80)
hours_studied = st.slider("Hours Studied per Week", 0, 20, 5)
social_interactions = st.slider("Social Interactions (0-10)", 0, 10, 5)
stress_score = st.slider("Reported Stress Score (0-10)", 0, 10, 5)

if st.button("Predict"):
    # Prepare input
    input_data = np.array([[grades, attendance, hours_studied, social_interactions, stress_score]])
    input_scaled = scaler.transform(input_data)
    
    # Predictions
    dropout_risk = dropout_model.predict_proba(input_scaled)[0][1]  # Probability of dropout
    predicted_stress = stress_model.predict(input_scaled)[0]
    
    st.write(f"Dropout Risk: {dropout_risk:.2f} (1 = High Risk)")
    st.write(f"Predicted Mental Stress Level: {predicted_stress:.2f}/10")
    
    # Alerts
    if dropout_risk > 0.7:
        st.warning("High dropout risk! Recommend intervention.")
    if predicted_stress > 7:
        st.error("High stress detected! Suggest counseling.")
    
    # Intervention Recommendations
    st.subheader("Recommended Interventions")
    recommendations = []
    
    if dropout_risk > 0.7:
        if grades < 60:
            recommendations.append("Academic Tutoring: Provide one-on-one sessions to improve grades.")
        if attendance < 70:
            recommendations.append("Attendance Monitoring: Schedule check-ins to boost participation.")
        if hours_studied < 4:
            recommendations.append("Study Skills Workshop: Offer time management and study techniques.")
    
    if predicted_stress > 7:
        recommendations.append("Mental Health Counseling: Refer to a counselor for stress management sessions.")
        if social_interactions < 4:
            recommendations.append("Peer Support Group: Encourage joining clubs or groups for social engagement.")
    
    if stress_score > 8:  # Based on input stress
        recommendations.append("Wellness Program: Suggest mindfulness apps or relaxation exercises.")
    
    if not recommendations:
        recommendations.append("No immediate interventions needed—monitor closely.")
    
    for rec in recommendations:
        st.info(rec)