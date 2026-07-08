"""
Student Placement Prediction — Streamlit App
Loads the trained pipeline (model.pkl) and predicts placement status +
probability score from user-entered student details.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    meta = joblib.load("feature_meta.pkl")
    return model, meta


model, meta = load_artifacts()

st.title("🎓 Student Placement Prediction")
st.write(
    "Enter a student's academic and skill profile below to predict whether "
    "they are likely to be **Placed** or **Not Placed**, along with a "
    "confidence score."
)

STATES = ["Delhi", "Gujarat", "Rajasthan", "Madhya Pradesh", "Uttar Pradesh",
          "Karnataka", "Tamil Nadu", "Maharashtra"]
CITIES = ["Jaipur", "Bhopal", "Bengaluru", "Delhi", "Chennai", "Indore",
          "Mumbai", "Lucknow", "Ahmedabad", "Pune"]
COLLEGE_TYPES = ["Private", "Autonomous", "Government"]
DEGREE_STREAMS = ["BCA", "B.Sc", "MCA", "BBA", "B.Com", "MBA", "B.Tech"]

with st.form("student_form"):
    st.subheader("Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=17, max_value=30, value=21)
        state = st.selectbox("State", STATES)
        city = st.selectbox("City", CITIES)
    with col2:
        college_type = st.selectbox("College Type", COLLEGE_TYPES)
        degree_stream = st.selectbox("Degree Stream", DEGREE_STREAMS)
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        internship_months = st.slider("Internship Months", 0, 12, 0)

    st.subheader("Academics")
    col3, col4, col5 = st.columns(3)
    with col3:
        ssc = st.slider("10th Percentage", 0, 100, 75)
        hsc = st.slider("12th Percentage", 0, 100, 75)
    with col4:
        grad = st.slider("Graduation Percentage", 0, 100, 70)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.01)
    with col5:
        backlogs = st.number_input("Backlogs", min_value=0, max_value=15, value=0)
        attendance = st.slider("Attendance (%)", 0, 100, 80)

    st.subheader("Skills & Activities")
    col6, col7, col8 = st.columns(3)
    with col6:
        projects = st.number_input("Projects", min_value=0, max_value=15, value=3)
        certifications = st.number_input("Certifications", min_value=0, max_value=15, value=2)
    with col7:
        aptitude = st.slider("Aptitude Score", 0, 100, 70)
        coding = st.slider("Coding Score", 0, 100, 70)
        technical = st.slider("Technical Score", 0, 100, 70)
    with col8:
        communication = st.slider("Communication Score", 0, 100, 70)
        mock_interview = st.slider("Mock Interview Score", 0, 100, 70)
        resume = st.slider("Resume Score", 0, 100, 70)

    submitted = st.form_submit_button("Predict Placement", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "State": state,
        "City": city,
        "College_Type": college_type,
        "Degree_Stream": degree_stream,
        "10th_Percentage": ssc,
        "12th_Percentage": hsc,
        "Graduation_Percentage": grad,
        "CGPA": cgpa,
        "Backlogs": backlogs,
        "Attendance": attendance,
        "Internship": internship,
        "Internship_Months": internship_months,
        "Projects": projects,
        "Certifications": certifications,
        "Aptitude_Score": aptitude,
        "Coding_Score": coding,
        "Communication_Score": communication,
        "Technical_Score": technical,
        "Mock_Interview_Score": mock_interview,
        "Resume_Score": resume,
    }])

    # Engineered features — must match training-time feature engineering
    row["Avg_Academic_Score"] = row[["10th_Percentage", "12th_Percentage", "Graduation_Percentage"]].mean(axis=1)
    row["Avg_Test_Score"] = row[["Aptitude_Score", "Coding_Score", "Technical_Score"]].mean(axis=1)
    row["Soft_Skill_Score"] = row[["Communication_Score", "Mock_Interview_Score", "Resume_Score"]].mean(axis=1)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"### ✅ Predicted: Placed")
    else:
        st.error(f"### ❌ Predicted: Not Placed")

    st.metric("Placement Probability", f"{probability * 100:.1f}%")
    st.progress(float(probability))

    with st.expander("What drove this prediction?"):
        st.write(
            "This model weighs academic scores (CGPA, percentages), test/coding "
            "performance, backlogs, and soft-skill scores most heavily. Higher "
            "scores and fewer backlogs push the prediction toward **Placed**."
        )

st.divider()
st.caption(
    "Model: Logistic Regression (tuned via GridSearchCV) trained on 10,000 "
    "student records. See the accompanying Jupyter notebook for full EDA, "
    "model comparison, and evaluation metrics."
)
