"""
app.py
======
Streamlit app for the Mwanza Mathematics Performance Predictive
Model.

The user enters:
    - Teacher-to-Student Ratio
    - School Type (Government / Private)
    - Mock Exam Grade (A, B, C, D, F)

After clicking "Enter", the app shows:
    - The predicted NECTA result (Pass / Fail)
    - The probability of passing / failing
    - Personalised suggestions based on the prediction

Run with:
    streamlit run app.py

Requires "model_artifacts.pkl" in the same folder, which is
created by running `python train_model.py` once.
"""

import os
import pandas as pd
import streamlit as st
import joblib

MODEL_FILE = "model_artifacts.pkl"

st.set_page_config(
    page_title="NECTA Mathematics Performance Predictor",
    page_icon="📊",
    layout="centered",
)


# --------------------------------------------------------------
# Load the trained model + encoders (cached so it loads once)
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


artifacts = load_artifacts()

st.title("Mwanza Mathematics Performance Predictor")
st.write(
    "Enter a student's details below to predict whether they will "
    "**Pass** or **Fail** the NECTA Form Four Mathematics examination, "
    "and to receive personalised suggestions."
)

if artifacts is None:
    st.error(
        f"Model file **'{MODEL_FILE}'** was not found.\n\n"
        "Please run the following command first (with "
        "`Mwanza_maths_updated.csv` in the same folder) to train "
        "the model and create this file:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

model = artifacts["model"]
model_name = artifacts.get("model_name", "Unknown Model")
le_school = artifacts["le_school"]
oe_mock = artifacts["oe_mock"]
feature_cols = artifacts["feature_cols"]
mock_order = artifacts["mock_order"]
accuracy = artifacts["accuracy"]

with st.sidebar:
    st.header("Model Information")
    st.write(f"**Model in use:** {model_name}")
    st.write(f"**Test accuracy:** {accuracy * 100:.2f}%")
    st.caption(
        "The model with the higher test accuracy "
        "(Logistic Regression or Random Forest) was chosen "
        "automatically during training."
    )

st.markdown("---")
st.subheader("Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    teacher_student_ratio = st.number_input(
        "Teacher-to-Student Ratio (1 : N)",
        min_value=1,
        max_value=300,
        value=45,
        step=1,
        help="Number of students per teacher, e.g. 45 means 1 teacher per 45 students.",
    )

    school_type = st.selectbox(
        "School Type",
        options=list(le_school.classes_),
    )

with col2:
    mock_grade = st.selectbox(
        "Mock Exam Grade",
        options=mock_order,
        index=0,
        help="Grade obtained in the mock examination (A is best, F is worst).",
    )

st.markdown("")
predict_clicked = st.button("Enter", type="primary", use_container_width=True)

# --------------------------------------------------------------
# Run prediction when the user clicks "Enter"
# --------------------------------------------------------------
if predict_clicked:
    # Encode inputs using the SAME encoders used during training
    school_encoded = int(le_school.transform([school_type])[0])
    mock_encoded = int(oe_mock.transform([[mock_grade]])[0][0])

    student_features = pd.DataFrame(
        [[teacher_student_ratio, school_encoded, mock_encoded]],
        columns=feature_cols,
    )

    prediction = model.predict(student_features)[0]
    probability = model.predict_proba(student_features)[0]  # [P(fail), P(pass)]
    prob_fail, prob_pass = float(probability[0]), float(probability[1])

    st.markdown("---")
    st.subheader("Prediction Result")

    summary_cols = st.columns(3)
    summary_cols[0].metric("School Type", school_type)
    summary_cols[1].metric("Teacher-to-Student Ratio", f"1 : {int(teacher_student_ratio)}")
    summary_cols[2].metric("Mock Exam Grade", mock_grade)

    if prediction == 1:
        st.success(f" PREDICTION: PASS  (Model used: {model_name})")
    else:
        st.error(f" PREDICTION: FAIL  (Model used: {model_name})")

    prob_cols = st.columns(2)
    prob_cols[0].metric("Probability of Pass", f"{prob_pass * 100:.1f}%")
    prob_cols[1].metric("Probability of Fail", f"{prob_fail * 100:.1f}%")
    st.progress(prob_pass)

    # ----------------------------------------------------------
    # Suggestions (mirrors the get_suggestions() logic from the
    # original notebook)
    # ----------------------------------------------------------
    st.markdown("---")
    st.subheader("Suggestions")

    if prediction == 0:
        if prob_pass >= 0.40:
            st.warning(
                "The student is at **MODERATE** risk of failing — "
                "targeted support can turn this around."
            )
        else:
            st.error(
                "The student is at **HIGH** risk of failing — "
                "urgent intervention is needed."
            )

        st.markdown(
            "**Suggestions to improve performance:**\n"
            "1. Enrol in remedial Mathematics classes focusing on weak topic areas.\n"
            "2. Teachers should use group assignments to keep large classes engaged.\n"
            "3. School should create a textbook-sharing or library rotation system.\n"
            "4. Student should practise past NECTA papers under timed conditions.\n"
            "5. Parents/guardians should be informed and support a structured home-study plan."
        )
    else:
        if prob_pass >= 0.80:
            st.success(
                "The student is **highly likely to pass** — "
                "keep up the excellent work!"
            )
        else:
            st.info(
                "The student is **likely to pass** — "
                "maintain current effort to stay on track."
            )

        st.markdown(
            "**Suggestions to maintain and improve performance:**\n"
            "1. Keep up current study discipline and avoid overconfidence.\n"
            "2. Engage in peer tutoring to reinforce personal understanding.\n"
            "3. Teachers should track progress through regular short tests.\n"
            "4. Attempt advanced NECTA questions to maximise the final grade.\n"
            "5. Maintain good health and sleep habits during the exam period."
        )
