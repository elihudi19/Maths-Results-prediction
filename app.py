import os
import io
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

st.set_page_config(
    page_title="NECTA Mathematics Performance Predictor",
    page_icon="📊",
    layout="centered",
)

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
model_name   = artifacts.get("model_name", "Unknown Model")
school_map   = artifacts.get("school_map", {"Government": 0, "Private": 1})
oe_mock      = artifacts["oe_mock"]
feature_cols = artifacts["feature_cols"]
mock_order   = artifacts["mock_order"]
accuracy     = artifacts["accuracy"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Test accuracy:** {accuracy * 100:.2f}%")

    st.subheader("TRAINED DATASET INFORMATION ABOUT PASS & FAIL")
    st.markdown(
        """
**PASS**
*Pass Grades in NECTA are A, B, C and D which are termed as 1*

**FAIL**
*Fail Grade in NECTA is only F which is termed as 0*
        """
    )

    st.markdown("---")
    st.subheader("Feature Coefficients")

    coef      = model.coef_[0]
    intercept = model.intercept_[0]

    coef_df = pd.DataFrame({
        "Feature": ["Teacher-to-Student Ratio", "School Type", "Mock Exam Grade"],
        "Coefficient": [round(coef[0], 4), round(coef[1], 4), round(coef[2], 4)],
    })
    st.dataframe(coef_df, hide_index=True, use_container_width=True)
    st.write(f"**Intercept:** {intercept:.4f}")

    st.markdown("---")
    st.subheader("INTERPRETATIONS")
    st.markdown(
        f"""
**Intercept: {intercept:.4f}**
*This means the probability to pass is starting at* **{1/(1+np.exp(-intercept)):.5f}**
*when all variables are zero.
Where in School Type; Private = 0 and in Mock Result; F = 0.
This occurred during the encoding process. In real world interpretation,
teacher-to-student ratio can never be zero.*
        """
    )
    st.markdown(
        f"""
**Teacher-to-Student Ratio** `{coef[0]:.4f}`
A larger class reduces the log-odds of passing slightly —
each additional student per teacher makes it marginally harder to pass.

**School Type** `{coef[1]:.4f}`
**Private** = *0 (Baseline)*, **Government** = *1*
A negative coefficient means Government school students have
**lower** log-odds of passing compared to Private school students,
all other variables remaining the same.

**Mock Exam Grade** `{coef[2]:.4f}`
The strongest predictor. Each grade step up (F→D→C→B→A)
substantially increases the log-odds of passing NECTA —
a student who scores A in the mock is far more likely to pass.
        """
    )

    st.markdown("---")
    st.subheader("DEVELOPERS INFORMATION")
    st.markdown(
        """
**ELIHUDI T ELIAMINI**
*CONTACT: 0756710637*

**ERENEST D MANYAMA**
*CONTACT: +255 682 436 629*
        """
    )

# ── Input form ───────────────────────────────────────────────────────────────
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
        help="Number of students per teacher.",
    )
    school_type = st.selectbox(
        "School Type",
        options=["Private", "Government"],
    )

with col2:
    mock_grade = st.selectbox(
        "Mock Exam Grade",
        options=mock_order,
        help="Grade in the mock exam (A is best, F is worst).",
    )

st.markdown("")
predict_clicked = st.button("Enter", type="primary", use_container_width=True)


# ── PDF generation ────────────────────────────────────────────────────────────
def generate_pdf(school_type, ratio, mock_grade, model_name,
                 prediction, prob_pass, prob_fail, message, suggestions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles   = getSampleStyleSheet()
    BLUE     = colors.HexColor("#1F4E79")
    GREEN    = colors.HexColor("#2ecc71")
    RED      = colors.HexColor("#e74c3c")
    LGRAY    = colors.HexColor("#f2f2f2")

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
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    story = [
        Paragraph("NECTA Mathematics Performance Prediction", title_s),
        Paragraph(f"Generated on {now}", sub_s),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10),
        Spacer(1, 0.2*cm),

        Paragraph("Student Input Summary", head_s),
        Table(
            [
                ["School Type",               school_type],
                ["Teacher-to-Student Ratio",  f"1 : {int(ratio)}"],
                ["Mock Exam Grade",           mock_grade],
                ["Model Used",                model_name],
            ],
            colWidths=[7*cm, 9*cm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (0,-1), LGRAY),
                ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 11),
                ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.white, LGRAY]),
                ("BOX",           (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ])
        ),
        Spacer(1, 0.3*cm),

        Paragraph("Prediction Result", head_s),
        Table(
            [
                ["PREDICTED OUTCOME",  result_label],
                ["Probability of Pass", f"{prob_pass * 100:.2f}%"],
                ["Probability of Fail", f"{prob_fail * 100:.2f}%"],
            ],
            colWidths=[7*cm, 9*cm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), result_color),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTNAME",      (0,1), (-1,-1),"Helvetica"),
                ("FONTSIZE",      (0,0), (-1,-1), 12),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, LGRAY]),
                ("BOX",           (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("TOPPADDING",    (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
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
        Paragraph(
            "Mwanza Mathematics Performance Prediction System — 2026",
            sub_s
        ),
        Paragraph(
            "Developers: ELIHUDI T ELIAMINI (0756710637) | "
            "ERENEST D MANYAMA (+255 682 436 629)",
            sub_s
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    if school_type in school_map:
        school_encoded = school_map[school_type]
    else:
        school_key = next(
            (k for k in school_map if k.lower() == school_type.lower()), None
        )
        if school_key is None:
            st.error(
                f"School type '{school_type}' not recognised. "
                f"Available: {list(school_map.keys())}"
            )
            st.stop()
        school_encoded = school_map[school_key]

    mock_encoded = int(
        oe_mock.transform(
            pd.DataFrame([[mock_grade]], columns=["mock_result"])
        )[0][0]
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
        st.success(f"PREDICTION: PASS   (Model: {model_name})")
    else:
        st.error(f"PREDICTION: FAIL   (Model: {model_name})")

    p1, p2 = st.columns(2)
    p1.metric("Probability of Pass", f"{prob_pass:.5f}")
    p2.metric("Probability of Fail", f"{prob_fail:.5f}")
    st.progress(prob_pass)

    # ── Suggestions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SUGGESTIONS")

    grade_colors = {
        "A": "rgba(0, 208, 132, 0.3)",
        "B": "rgba(0, 153, 255, 0.3)",
        "C": "#FFA500",
        "D": "rgba(255, 215, 0, 0.3)",
        "F": "rgba(255, 68, 68, 0.3)",
    }
    color_hex = grade_colors.get(mock_grade, "rgba(128,128,128,0.3)")

    if prediction == 0:
        message = (
            "The student is at <b>HIGH</b> risk of failing; "
            "urgent intervention is needed."
        )
        suggestion_lines = [
            "1. Enrol in remedial Mathematics classes focusing on weak topic areas.",
            "2. Teachers should use group assignments to keep large classes engaged.",
            "3. School should create a textbook-sharing or library rotation system.",
            "4. Student should practise past NECTA papers under timed conditions.",
            "5. Parents/guardians should be informed and support a structured home-study plan.",
        ]
        pdf_header = "Suggestions to improve performance:"
    else:
        if mock_grade == "A":
            message = "Great job! Your hard work really shows. Keep it up!"
        elif mock_grade == "B":
            message = "Good job! You did very well — keep pushing for the top."
        elif mock_grade == "C":
            message = "Nice work! You passed, but you can do even better next time."
        else:
            message = "You passed! It was close, so let's study harder next time."
        suggestion_lines = [
            "1. Keep up current study discipline and avoid overconfidence.",
            "2. Engage in peer tutoring to reinforce personal understanding.",
            "3. Teachers should track progress through regular short tests.",
            "4. Attempt advanced NECTA questions to maximise the final grade.",
            "5. Maintain good health and sleep habits during the exam period.",
        ]
        pdf_header = "Suggestions to maintain and improve performance:"

    suggestions_html = "<br>".join(suggestion_lines)
    st.markdown(f"""
    <div style="background-color:{color_hex}; padding:15px; border-radius:10px;
                border-left:5px solid {color_hex};">
        <h4 style="color:black; margin-top:0;">STUDENT'S MESSAGE</h4>
        <p style="color:black; font-size:16px; margin-bottom:0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color:{color_hex}; padding:15px; border-radius:10px;
                border-left:5px solid {color_hex}; margin-top:15px;">
        <p style="color:black; font-size:14px; line-height:1.8; margin:0;">
            <b>{pdf_header}</b><br>{suggestions_html}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    plain_message = message.replace("<b>", "").replace("</b>", "")

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
    )

    st.download_button(
        label="⬇️ Download Prediction & Suggestions as PDF",
        data=pdf_buffer,
        file_name=f"NECTA_Prediction_{school_type}_{mock_grade}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
