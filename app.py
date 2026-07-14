"""
app.py
======
Streamlit app for the Mwanza Mathematics Performance Predictor.

Run ONCE before launching:
    python train_model.py

Then launch the app:
    streamlit run app.py
"""

import io
import os
import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Constants ────────────────────────────────────────────────────────────[[...]
MODEL_FILE  = "model_artifacts.pkl"
MOCK_ORDER  = ["A", "B", "C", "D", "F"]   # best → worst (display order)
SCHOOL_MAP  = {"Government": 1, "Private": 0}

# ── Page config ──────────────────────────────────────────────────────────[...]
st.set_page_config(
    page_title="NECTA Mathematics Performance Predictor",
    page_icon="📊",
    layout="centered",
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

artifacts = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────[...[...]
st.title("📊NECTA Mathematics Performance Predictor")
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
model_name   = artifacts.get("model_name", "Unknown Model")
oe_mock      = artifacts["oe_mock"]
feature_cols = artifacts["feature_cols"]
accuracy     = artifacts["accuracy"]

# ── Sidebar ──────────────────────────────────────────────────────────[...]
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Test accuracy:** {accuracy * 100:.2f}%")

    st.markdown("---")
    st.subheader("TRAINED DATASET: PASS & FAIL")
    st.markdown(
        """
**PASS**
*Grades A, B, C and D → coded as 1*

**FAIL**
*Grade F only → coded as 0*
        """
    )

    st.markdown("---")
    if hasattr(model, "coef_"):
        st.subheader("Feature Coefficients")
        coef      = model.coef_[0]
        intercept = model.intercept_[0]
        coef_df = pd.DataFrame({
            "Feature":     ["Teacher-to-Student Ratio", "School Type", "Mock Exam Grade"],
            "Coefficient": [round(c, 4) for c in coef],
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
        st.write(f"**Intercept:** {intercept:.4f}")

        st.markdown("---")
        st.subheader("INTERPRETATIONS")
        st.markdown(
            f"""
**Intercept: {intercept:.4f}**  
This means probability of student to pass is starting **0.10724** When all variable are zero.
School Type:**Private = 0** and Mock Grade:**F = 0** During Encoding Process and Teacher to student ratio **= 0**.
But in realy world interpretations Teacher to student ratio cannot be zero.

**Teacher-to-Student Ratio** `{coef[0]:.4f}`  
A larger class reduces the log-odds of passing slightly -
each additional student per teacher makes it marginally harder to pass.

**School Type** `{coef[1]:.4f}`   
**Private = 0 (Baseline)**, **Government = 1**
A negative coefficient means Government school students have
**lower** log-odds of passing compared to Private school students,
all other variables being equal.

**Mock Exam Grade** `{coef[2]:.4f}`  
The strongest predictor. Each grade step up (F→D→C→B→A)
substantially increases the log-odds of passing NECTA.
            """
        )

    st.markdown("---")
    st.subheader("DEVELOPER INFORMATION")
    st.markdown(
        """
**ELIHUDI T ELIAMINI**  
*Contact: 0756710637*

**ERENEST D MANYAMA**   
*Contact: +255 682 436 629*
        """
    )

# ── Input form ──────────────────────────────────────────────────────────[...]
st.markdown("---")
st.subheader("Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    teacher_student_ratio = st.number_input(
        "Teacher-to-Student Ratio (1 : N)",
        step=1,
        help="Number of students per teacher.",
    )
    school_type = st.selectbox(
        "School Type",
        options=["Private", "Government"],
    )

    mock_grade = st.selectbox(
        "Mock Exam Grade",
        options=["F", "D", "C", "B", "A"],
        help="Grade obtained in the mock exam (A is best, F is worst).",
    )
st.markdown("")
predict_clicked = st.button("**PREDICT**", type="primary", use_container_width=True)


# ── PDF generation ────────────────────────────────────────��─────────────────[.[...]
def generate_pdf(school_type, ratio, mock_grade, model_name,
                 prediction, prob_pass, prob_fail, message, suggestions, message_color):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    BLUE  = colors.HexColor("#1F4E79")
    GREEN = colors.HexColor("#2ecc71")
    RED   = colors.HexColor("#e74c3c")
    LGRAY = colors.HexColor("#f2f2f2")

    title_s = ParagraphStyle("TitleS", parent=styles["Title"],
                             fontSize=17, textColor=BLUE,
                             spaceAfter=4, alignment=TA_CENTER)
    sub_s   = ParagraphStyle("SubS", parent=styles["Normal"],
                             fontSize=10, textColor=colors.gray,
                             spaceAfter=4, alignment=TA_CENTER)
    head_s  = ParagraphStyle("HeadS", parent=styles["Heading2"],
                             fontSize=13, textColor=BLUE,
                             spaceBefore=12, spaceAfter=4)
    body_s  = ParagraphStyle("BodyS", parent=styles["Normal"],
                             fontSize=11, leading=17, spaceAfter=4,
                             alignment=TA_JUSTIFY)

    result_label = "PASS" if prediction == 1 else "FAIL"
    result_color = GREEN  if prediction == 1 else RED
    eat_tz = ZoneInfo("Africa/Nairobi")
    now    = datetime.datetime.now(eat_tz).strftime("%d %B %Y, %H:%M")

    # Build result table with only the relevant probability
    if prediction == 1:
        result_table_data = [
            ["PREDICTED OUTCOME",   result_label],
            ["Probability of Pass", f"{prob_pass:.5f}"],
        ]
    else:
        result_table_data = [
            ["PREDICTED OUTCOME",   result_label],
            ["Probability of Fail", f"{prob_fail:.5f}"],
        ]

    story = [
        Paragraph("NECTA Mathematics Performance Prediction", title_s),
        Paragraph(f"Generated on {now} (EAT)", sub_s),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10),
        Spacer(1, 0.2*cm),

        Paragraph("Student Input Summary", head_s),
        Table(
            [
                ["School Type",              school_type],
                ["Teacher-to-Student Ratio", f"1 : {int(ratio)}"],
                ["Mock Exam Grade",          mock_grade],
                ["Model Used",               model_name],
            ],
            colWidths=[7*cm, 9*cm],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (0, -1), LGRAY),
                ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LGRAY]),
                ("BOX",           (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ])
        ),
        Spacer(1, 0.3*cm),

        Paragraph("Prediction Result", head_s),
        Table(
            result_table_data,
            colWidths=[7*cm, 9*cm],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), result_color),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",      (0, 0), (-1, -1), 12),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGRAY]),
                ("BOX",           (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("TOPPADDING",    (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ])
        ),
        Spacer(1, 0.3*cm),

        Paragraph("Student's Message", head_s),
        Paragraph(message, body_s),
        Spacer(1, 0.2*cm),

        Paragraph("Suggestions", head_s),
        HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6),
    ]

    for line in suggestions:
        story.append(Paragraph(line, body_s))

    story += [
        Spacer(1, 0.6*cm),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=6),
        Paragraph("Mwanza Mathematics Performance Prediction System-2026", sub_s),
        Paragraph(
            "Developers: ELIHUDI T ELIAMINI (0756710637) | "
            "ERENEST D MANYAMA (+255 682 436 629)",
            sub_s,
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── Helper function to determine message and color based on probability ──────
def get_message_and_color(prob_pass):
    """
    Determine student message and background color based on probability of pass.
    
    Probability ranges:
    - 0.7 to 1.0: Good job! Maintain a progress
    - 0.5 to 0.69: Study hard to maintain Progress
    - 0.0 to 0.49: You are at risk, Study hard.
    """
    if prob_pass >= 0.7:
        message = "Good job! Maintain a progress"
        color_hex = "rgba(0, 208, 132, 0.3)"  # Green
        pdf_color = colors.HexColor("#00D084")  # Solid green for PDF
        suggestions_header = "Suggestions to maintain and improve performance:"
        suggestion_lines = [
            "1. Keep up current study discipline and avoid overconfidence.",
            "2. Engage in peer tutoring to reinforce personal understanding.",
            "3. Teachers should track progress through regular short tests.",
            "4. Attempt advanced NECTA questions to maximise the final grade.",
            "5. Maintain good health and sleep habits during the exam period.",
        ]
    elif prob_pass >= 0.5:
        message = "Study hard to maintain Progress"
        color_hex = "rgba(255, 165, 0, 0.3)"  # Orange
        pdf_color = colors.HexColor("#FFA500")  # Solid orange for PDF
        suggestions_header = "Suggestions to improve and maintain performance:"
        suggestion_lines = [
            "1. Focus on understanding weak concept areas in Mathematics.",
            "2. Increase study time and consistency in revision.",
            "3. Work with classmates on challenging topics through group study.",
            "4. Practice past NECTA papers regularly under timed conditions.",
            "5. Seek help from teachers for areas of difficulty.",
        ]
    else:
        message = "You are at risk, Study hard."
        color_hex = "rgba(255, 68, 68, 0.3)"  # Red
        pdf_color = colors.HexColor("#FF4444")  # Solid red for PDF
        suggestions_header = "Suggestions to improve performance:"
        suggestion_lines = [
            "1. Enrol in remedial Mathematics classes focusing on weak topic areas.",
            "2. Teachers should use group assignments to keep large classes engaged.",
            "3. School should create a textbook-sharing or library rotation system.",
            "4. Student should practise past NECTA papers under timed conditions.",
            "5. Parents/guardians should be informed and support a structured home-study plan.",
        ]
    
    return message, color_hex, suggestions_header, suggestion_lines, pdf_color


# ── Prediction ──────────────────────────────────────────────────────────[...]
if predict_clicked:
    school_encoded = SCHOOL_MAP[school_type]
    mock_encoded   = int(
        oe_mock.transform(pd.DataFrame([[mock_grade]], columns=["mock_result"]))[0][0]
    )

    X = pd.DataFrame(
        [[teacher_student_ratio, school_encoded, mock_encoded]],
        columns=feature_cols,
    )

    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    prob_fail, prob_pass = float(probability[0]), float(probability[1])

    st.markdown("---")
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("School Type",        school_type)
    c2.metric("Teacher : Student",  f"1 : {int(teacher_student_ratio)}")
    c3.metric("Mock Grade",         mock_grade)

    if prediction == 1:
        st.success(f"PREDICTION: **PASS**   (Model: {model_name})")
    else:
        st.error(f"PREDICTION: **FAIL**   (Model: {model_name})")

    # Display only the relevant probability based on prediction
    if prediction == 1:
        st.metric("Probability of Pass", f"{prob_pass:.5f}")
        st.progress(prob_pass)
    else:
        st.metric("Probability of Fail", f"{prob_fail:.5f}")
        st.progress(prob_fail)

    # ── Suggestions ────────────────────────────────────────────────────────[...]
    st.markdown("---")
    st.subheader("SUGGESTIONS")

    # Get message, color, and suggestions based on probability score
    message, color_hex, pdf_header, suggestion_lines, pdf_color = get_message_and_color(prob_pass)

    suggestions_html = "<br>".join(suggestion_lines)

    st.markdown(
        f"""
        <div style="background-color:{color_hex}; padding:15px; border-radius:10px;
                    border-left:5px solid gray;">
            <h4 style="color:black; margin-top:0;">STUDENT'S MESSAGE</h4>
            <p style="color:black; font-size:16px; margin-bottom:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background-color:{color_hex}; padding:15px; border-radius:10px;
                    border-left:5px solid gray; margin-top:15px;">
            <p style="color:black; font-size:14px; line-height:1.8; margin:0;">
                <b>{pdf_header}</b><br>{suggestions_html}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    plain_message = message

    pdf_buffer = generate_pdf(
        school_type=school_type,
        ratio=teacher_student_ratio,
        mock_grade=mock_grade,
        model_name=model_name,
        prediction=prediction,
        prob_pass=prob_pass,
        prob_fail=prob_fail,
        message=plain_message,
        suggestions=[pdf_header] + suggestion_lines,
        message_color=pdf_color,
    )

    st.download_button(
        label="⬇️  Download Prediction & Suggestions as PDF",
        data=pdf_buffer,
        file_name=f"NECTA_Prediction_{school_type}_{mock_grade}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
