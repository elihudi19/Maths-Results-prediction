# Mwanza Mathematics Performance Predictor — Streamlit App

This app is based on your notebook
`MATHEMATICS_PERFOMANCE_PREDECTION__1_.ipynb`. It predicts whether a
student will **Pass** or **Fail** the NECTA Form Four Mathematics
examination from:

- Teacher-to-Student Ratio
- School Type (Government / Private)
- Mock Exam Grade (A, B, C, D, F)

...and shows the same personalised suggestions as the notebook's
`get_suggestions()` function.

## Files

| File                  | Purpose                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `train_model.py`      | Loads `Mwanza_maths_updated.csv`, trains Logistic Regression & Random Forest (same as the notebook), picks the best one, and saves it + encoders to `model_artifacts.pkl`. |
| `app.py`               | The Streamlit app. Loads `model_artifacts.pkl` and lets users enter student data, then click **Enter** to see the prediction and suggestions. |
| `requirements.txt`    | Python packages needed to run everything.                                |

## Setup

1. Put your dataset `Mwanza_maths_updated.csv` in this same folder
   (it must contain the columns `teacher_to_student_ratio`,
   `school_type`, `mock_result`, `NECTA_result`).

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (run this once, or again whenever your data
   changes):

   ```bash
   python train_model.py
   ```

   This creates `model_artifacts.pkl` in the same folder — the
   Streamlit app needs this file to run.

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

   This opens the app in your browser. Enter the student's data
   and click **Enter** to see the prediction and suggestions.

## Notes

- The encoders (`LabelEncoder` for school type, `OrdinalEncoder` for
  mock grade) are saved together with the model, so the app encodes
  new inputs exactly the same way the notebook did.
- The sidebar shows which model (Logistic Regression or Random
  Forest) was selected as the best one, and its test accuracy.
- If you retrain on new data, just re-run `train_model.py` — the app
  will automatically pick up the new `model_artifacts.pkl` the next
  time it is started (or refresh the page after restarting the app).
