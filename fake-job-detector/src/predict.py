import joblib

# Load the trained model
model = joblib.load('models/job_detector.pkl')

# Sample input — you can change this to test others
title = input("Enter job title: ")
description = input("Enter job description: ")

# Combine title and description as done during training
input_text = title + " " + description

# Predict
prediction = model.predict([input_text])[0]

if prediction == 1:
    print("⚠️ This job posting might be FAKE!")
else:
    print("✅ This looks like a REAL job posting.")
