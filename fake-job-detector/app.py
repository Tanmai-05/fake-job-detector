import streamlit as st
import joblib

# Load trained model
model = joblib.load('models/job_detector.pkl')

# UI
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ Fake Job Posting Detector")
st.write("Paste the job title and description to check if it's real or fake.")

# Input fields
title = st.text_input("Enter Job Title")
description = st.text_area("Enter Job Description", height=200)

if st.button("Predict"):
    if not title or not description:
        st.warning("Please enter both title and description.")
    else:
        input_text = title + " " + description
        prediction = model.predict([input_text])[0]
        if prediction == 1:
            st.error("⚠️ This job posting might be FAKE!")
        else:
            st.success("✅ This looks like a REAL job posting.")
