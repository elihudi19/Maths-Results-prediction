import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

MODEL_FILE = "model_artifacts.pkl"

st.set_page_config(
    page_title="NECTA Mathematics Performance Predictor",
    page_icon="📊",
    layout="centered",
)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

artifacts = load_artifacts()

st.title("📊 Mathematics Performance Predictor")
st.write(
    "Enter a student's details below to predict whether they will "
    "**Pass** or **Fail** the NECTA Form Four Mathematics examination."
)

if artifacts is None:
    st.error(
        f"Model file **'{MODEL_FILE}'** was not found.\n\n"
        "Run `python train_model.py` first to generate it."
    )
    st.stop()

model        = artifacts["model"]
model_name = artifacts.get("model_name", "Unknown Model")
school_map   = artifacts.get("school_map", {"Government": 1, "Private": 0})       # {"Private": 0, "Government": 1}
oe_mock      = artifacts["oe_mock"]
feature_cols = artifacts["feature_cols"]
mock_order   = artifacts["mock_order"]
accuracy     = artifacts["accuracy"]

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Test accuracy:** {accuracy * 100:.2f}%")

    st.subheader("TRAINED DATASET INFORMATION ABOUT PASS & FAIL")
    st.markdown(
        """
**PASS**    
*Pass Grades in NECTA are A,B,C and D which are termed as 1*     

**FAIL**          
*Fail Grade in NECTA is only F which is termed as 0*
         """
    )
    
    st.markdown("---")
    st.subheader("Feature Coefficients")

    coef = model.coef_[0]          # shape (3,)
    intercept = model.intercept_[0]

    coef_df = pd.DataFrame({
        "Feature": ["Teacher-to-Student Ratio", "School Type", "Mock Exam Grade"],
        "Coefficient": [round(coef[0], 4), round(coef[1], 4), round(coef[2], 4)], 
        "This means the probability to pass is starting 0.107 when all variable is zero. But in real world enterpretation teacher to student ratio can never be zero",
    })
    st.dataframe(coef_df, hide_index=True, use_container_width=True)
    st.write(f"**Intercept:** {intercept:.4f}")

    st.markdown("---")
    st.subheader("INTERPRETATIONS")
    st.markdown(
        """
**Teacher-to-Student Ratio** `{:.4f}`  
A larger class reduces the log-odds of passing slightly -
each additional student per teacher makes it marginally harder to pass.

**School Type** `{:.4f}`  
A negative coefficient means Government school students have
**lower** log-odds of passing compared to Private school students,
all other variables remain the same.

**Mock Exam Grade** `{:.4f}`  
The strongest predictor. Each grade step up (F→D→C→B→A)
substantially increases the log-odds of passing NECTA -
a student who scores A in the mock is far more likely to pass.
        """.format(coef[0], coef[1], coef[2])
    )
    st.subheader("DEVELOPERS INFORMATION")
    st.markdown(
        """
**ELIHUDI T ELIAMINI**    
*CONTACT: 0756710637*     

**ERENEST D MANYAMA**          
*CONTACT: +255 682 436 629*
         """
    )

# ── Input form ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    teacher_student_ratio = st.number_input(
        "Teacher-to-Student Ratio (1 : N)", min_value=1, step=0,
        help="Number of students per teacher.",
    )
    school_type = st.selectbox(
        "School Type",
        options=["Government", "Private"],
    )

with col2:
    mock_grade = st.selectbox(
        "Mock Exam Grade",
        options=mock_order,
        help="Grade in the mock exam (A is best, F is worst).",
    )

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
predict_clicked = st.button("Enter", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────
if predict_clicked:
    # Safe key lookup with case-insensitive fallback
    if school_type in school_map:
        school_encoded = school_map[school_type]
    else:
        # Try case-insensitive match
        school_key = next((k for k in school_map.keys() if k.lower() == school_type.lower()), None)
        if school_key is None:
            st.error(f"School type '{school_type}' not found in model. Available options: {list(school_map.keys())}")
            st.stop()
        school_encoded = school_map[school_key]
    
    mock_encoded   = int(oe_mock.transform([[mock_grade]])[0][0])

    X = pd.DataFrame(
        [[teacher_student_ratio, school_encoded, mock_encoded]],
        columns=feature_cols,
    )

    prediction   = model.predict(X)[0]
    probability  = model.predict_proba(X)[0]
    prob_fail, prob_pass = float(probability[0]), float(probability[1])

    st.markdown("---")
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("School Type", school_type)
    c2.metric("Teacher : Student", f"1 : {int(teacher_student_ratio)}")
    c3.metric("Mock Grade", mock_grade)

    if prediction == 1:
        st.success(f"  PREDICTION: PASS   (Model: {model_name})")
    else:
        st.error(f"  PREDICTION: FAIL   (Model: {model_name})")

    p1, p2 = st.columns(2)
    p1.metric("Probability of Pass", f"{prob_pass :.6f}")
    p2.metric("Probability of Fail", f"{prob_fail :.6f}")
    st.progress(prob_pass)

    # ── Suggestions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SUGGESTIONS")

    # Determine grade level and color based on mock grade
    grade_colors = {
        "A": {"level": "STUDENT'S MESSAGE", "hex": "rgba(0, 208, 132, 0.3)"},
        "B": {"level": "STUDENT'S MESSAGE", "hex": "rgba(0, 153, 255, 0.3)"},
        "C": {"level": "STUDENT'S MESSAGE", "hex": "#FFA500"},
        "D": {"level": "STUDENT'S MESSAGE", "hex": "rgba(255, 215, 0, 0.3)"},
        "F": {"level": "STUDENT'S MESSAGE", "hex": "rgba(255, 68, 68, 0.3)"},
    }

    grade_info = grade_colors.get(mock_grade, {"level": "Unknown", "hex": "rgba(128, 128, 128, 0.3)"})
    grade_level = grade_info["level"]
    color_hex = grade_info["hex"]

    if prediction == 0:
        risk = "HIGH"
        
        st.markdown(f"""
        <div style="background-color: {color_hex}; padding: 15px; border-radius: 10px; border-left: 5px solid {color_hex};">
            <h4 style="color: black; margin-top: 0;">{grade_level}</h4>
            <p style="color: black; font-size: 16px; margin-bottom: 0;">
                The student is at <b>{"HIGH"}</b> risk of failing;
                {"urgent intervention is needed"}.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: {color_hex}; padding: 15px; border-radius: 10px; border-left: 5px solid {color_hex}; margin-top: 15px;">
            <p style="color: black; font-size: 14px; line-height: 1.8; margin: 0;">
                <b>Suggestions to improve performance:</b><br>
                1. Enrol in remedial Mathematics classes focusing on weak topic areas.<br>
                2. Teachers should use group assignments to keep large classes engaged.<br>
                3. School should create a textbook-sharing or library rotation system.<br>
                4. Student should practise past NECTA papers under timed conditions.<br>
                5. Parents/guardians should be informed and support a structured home-study plan.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Special handling for Grade C
        if mock_grade == "C":
            message = "Nice work! You passed, but you can do even better next time."
        elif mock_grade == "D":
            message = "You passed! It was close, so let's study harder next time"
        elif mock_grade == "B":
            message = "Good job! You did very well-keep pushing for the top"
        else:
            message = "Great job! Your hard work really shows. Keep it up!"
        
        st.markdown(f"""
        <div style="background-color: {color_hex}; padding: 15px; border-radius: 10px; border-left: 5px solid {color_hex};">
            <h4 style="color: black; margin-top: 0;">{grade_level}</h4>
            <p style="color: black; font-size: 16px; margin-bottom: 0;">
                {message}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: {color_hex}; padding: 15px; border-radius: 10px; border-left: 5px solid {color_hex}; margin-top: 15px;">
            <p style="color: black; font-size: 14px; line-height: 1.8; margin: 0;">
                <b>Suggestions to maintain and improve performance:</b><br>
                1. Keep up current study discipline and avoid overconfidence.<br>
                2. Engage in peer tutoring to reinforce personal understanding.<br>
                3. Teachers should track progress through regular short tests.<br>
                4. Attempt advanced NECTA questions to maximise the final grade.<br>
                5. Maintain good health and sleep habits during the exam period.
            </p>
        </div>
        """, unsafe_allow_html=True)
