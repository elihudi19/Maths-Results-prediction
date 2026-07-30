"""
app.py
======
Streamlit app for the Mwanza Mathematics Performance Predictor.

3-variable version (Teacher-to-Student Ratio, School Type, Mock Exam Grade)
with an AUTOMATIC, per-student personalised suggestion system — each
suggestion is generated dynamically from how much that student's own inputs
are pulling their prediction up or down (via the logistic regression
coefficients), instead of a single static message picked from a probability
bucket.

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

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_FILE  = "model_artifacts.pkl"
MOCK_ORDER  = ["A", "B", "C", "D", "F"]   # best → worst (display order)
SCHOOL_MAP  = {"Government": 1, "Private": 0}

FRIENDLY_NAMES = {
    "ratio":  "Teacher-to-Student Ratio",
    "school": "School Type",
    "mock":   "Mock Exam Grade",
}

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NECTA Mathematics Performance Predictor",
    page_icon="📊",
    layout="centered",
)

# ── Load model artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

artifacts = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────
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

HAS_COEF = hasattr(model, "coef_")

# ── Sidebar ────────────────────────────────────────────────────────────
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
    if HAS_COEF:
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

# ── Input form ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    teacher_student_ratio = st.number_input(
        "Teacher-to-Student Ratio (1 : N)",
        step=1,
        value=1,
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


# ── Automatic personalised-suggestion engine ────────────────────────────
def compute_contributions(ratio, school_encoded, mock_encoded, model):
    """
    Work out how much each of the 3 inputs is currently pushing THIS
    student's log-odds of passing up or down, relative to the best
    possible state for that variable. A negative contribution means the
    variable is currently a risk factor (pulling toward FAIL); a
    contribution of zero or higher means it's already at (or helping
    toward) its best state.

    - Teacher-to-Student Ratio: continuous. Best case is the smallest
      possible ratio (1 : 1), so contribution = (ratio - 1) * coef.
    - School Type: categorical (0/1). Best category is whichever
      encoding the coefficient's sign favours.
    - Mock Exam Grade: ordinal. Best category is grade "A", so
      contribution is scored relative to A's encoded value.
    """
    coef = model.coef_[0]
    ratio_coef, school_coef, mock_coef = coef[0], coef[1], coef[2]

    ratio_contribution = (ratio - 1) * ratio_coef

    best_school_value = 1.0 if school_coef > 0 else 0.0
    school_contribution = (school_encoded - best_school_value) * school_coef

    best_mock_encoded = int(
        oe_mock.transform(pd.DataFrame([["A"]], columns=["mock_result"]))[0][0]
    )
    mock_contribution = (mock_encoded - best_mock_encoded) * mock_coef

    return {
        "ratio":  ratio_contribution,
        "school": school_contribution,
        "mock":   mock_contribution,
    }


def suggestion_for(feature, raw_value):
    """Actionable suggestion for a factor currently working against the student."""
    table = {
        "ratio": (
            f"The teacher-to-student ratio (1:{int(raw_value)}) is high, which "
            "typically means less individual attention per student. Advocate for "
            "smaller class sizes, additional tutoring sessions, or peer study "
            "groups to help offset this."
        ),
        "school": (
            "This school type is associated with a lower average pass rate in the "
            "training data (often linked to fewer resources or larger classes). "
            "A structured personal timetable with dedicated study hours, plus "
            "resource-sharing with better-resourced schools, can help offset this."
        ),
        "mock": (
            f"The mock exam grade ({raw_value}) suggests the student is not yet "
            "fully prepared. Focus revision on the specific topics that were "
            "missed in the mock, and practise past NECTA papers under timed "
            "conditions."
        ),
    }
    return table.get(feature, "Review this factor with a teacher for tailored advice.")


def strength_note_for(feature, raw_value):
    """Positive reinforcement message for a factor already working in the student's favour."""
    table = {
        "ratio": (
            f"A teacher-to-student ratio of 1:{int(raw_value)} is favourable and "
            "supports the passing prediction."
        ),
        "school": (
            "This school type is associated with a stronger studying environment "
            "in the training data, which is currently working in the student's "
            "favour."
        ),
        "mock": (
            f"The mock exam grade ({raw_value}) is a strong positive signal — "
            "keep up this momentum leading into the final exam."
        ),
    }
    return table.get(feature, "This factor is currently helping.")


# ── PDF generation ───────────────────────────────────────────────────────
def generate_pdf(school_type, ratio, mock_grade, model_name,
                 prediction, prob_pass, prob_fail, risk_factors, positive_factors,
                 raw_values):
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
    factor_head_s = ParagraphStyle("FactorHeadS", parent=styles["Normal"],
                             fontSize=11, leading=17, spaceBefore=6,
                             fontName="Helvetica-Bold")

    result_label = "PASS" if prediction == 1 else "FAIL"
    result_color = GREEN  if prediction == 1 else RED
    eat_tz = ZoneInfo("Africa/Nairobi")
    now    = datetime.datetime.now(eat_tz).strftime("%d %B %Y, %H:%M")

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

        Paragraph("Personalised Suggestions", head_s),
        HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6),
    ]

    if not risk_factors:
        story.append(Paragraph(
            "All measured factors are currently working in this student's favour. "
            "Keep up the strong routine!", body_s
        ))
    else:
        story.append(Paragraph(
            "All factors below are currently reducing this student's chance of "
            "passing, ranked from most to least impactful:", body_s
        ))
        for i, (feat, contrib) in enumerate(risk_factors, start=1):
            friendly = FRIENDLY_NAMES.get(feat, feat)
            text = suggestion_for(feat, raw_values[feat])
            story.append(Paragraph(f"{i}. {friendly}", factor_head_s))
            story.append(Paragraph(text, body_s))

    if positive_factors:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Factors Already Working Well", head_s))
        for feat, contrib in positive_factors:
            friendly = FRIENDLY_NAMES.get(feat, feat)
            note = strength_note_for(feat, raw_values[feat])
            story.append(Paragraph(f"{friendly}: {note}", body_s))

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


# ── Prediction ───────────────────────────────────────────────────────────
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

    if prediction == 1:
        st.metric("Probability of Pass", f"{prob_pass:.5f}")
        st.progress(prob_pass)
    else:
        st.metric("Probability of Fail", f"{prob_fail:.5f}")
        st.progress(prob_fail)

    # ── Automatic personalised suggestions ───────────────────────────────
    st.markdown("---")
    st.subheader("PERSONALISED SUGGESTIONS")

    if not HAS_COEF:
        st.warning(
            f"The loaded model ('{model_name}') doesn't expose coefficients, so "
            "personalised, factor-by-factor suggestions can't be generated for it. "
            "Suggestions require a linear model such as Logistic Regression."
        )
        risk_factors, positive_factors = [], []
        raw_values = {}
    else:
        contributions = compute_contributions(
            teacher_student_ratio, school_encoded, mock_encoded, model
        )
        raw_values = {
            "ratio":  teacher_student_ratio,
            "school": school_type,
            "mock":   mock_grade,
        }

        sorted_factors   = sorted(contributions.items(), key=lambda kv: kv[1])
        risk_factors     = [f for f in sorted_factors if f[1] < 0]
        positive_factors = [f for f in sorted_factors if f[1] >= 0]

        overall_color = "rgba(0, 208, 132, 0.3)" if prob_pass >= 0.7 else (
            "rgba(255, 165, 0, 0.3)" if prob_pass >= 0.5 else "rgba(255, 68, 68, 0.3)"
        )

        if not risk_factors:
            risk_html = (
                "<p style='color:black; margin:0;'>All measured factors are "
                "currently working in this student's favour. Keep up the strong routine!</p>"
            )
        else:
            lines = []
            for i, (feat, contrib) in enumerate(risk_factors, start=1):
                friendly = FRIENDLY_NAMES.get(feat, feat)
                text = suggestion_for(feat, raw_values[feat])
                lines.append(f"<b>{i}. {friendly}</b><br>{text}")
            risk_html = (
                "<p style='color:black; margin:0 0 8px 0;'>All factors below are "
                "currently reducing this student's chance of passing, ranked from "
                "most to least impactful:</p>"
                + "<br><br>".join(lines)
            )

        st.markdown(
            f"""
            <div style="background-color:{overall_color}; padding:15px; border-radius:10px;
                        border-left:5px solid gray;">
                <h4 style="color:black; margin-top:0;">FACTORS TO WORK ON</h4>
                {risk_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if positive_factors:
            strength_lines = []
            for feat, contrib in positive_factors:
                friendly = FRIENDLY_NAMES.get(feat, feat)
                note = strength_note_for(feat, raw_values[feat])
                strength_lines.append(f"<b>{friendly}</b>: {note}")
            st.markdown(
                f"""
                <div style="background-color:rgba(0, 208, 132, 0.15); padding:15px; border-radius:10px;
                            border-left:5px solid gray; margin-top:15px;">
                    <h4 style="color:black; margin-top:0;">FACTORS ALREADY WORKING WELL</h4>
                    <p style="color:black; font-size:14px; line-height:1.8; margin:0;">
                        {"<br>".join(strength_lines)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    pdf_buffer = generate_pdf(
        school_type=school_type,
        ratio=teacher_student_ratio,
        mock_grade=mock_grade,
        model_name=model_name,
        prediction=prediction,
        prob_pass=prob_pass,
        prob_fail=prob_fail,
        risk_factors=risk_factors,
        positive_factors=positive_factors,
        raw_values=raw_values,
    )

    st.download_button(
        label="⬇️  Download Prediction & Suggestions as PDF",
        data=pdf_buffer,
        file_name=f"NECTA_Prediction_{school_type}_{mock_grade}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
