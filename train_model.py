"""
train_model.py
================
Trains the Mwanza Mathematics Performance Predictive Model
(Logistic Regression vs Random Forest), exactly as in the
original notebook, and saves the best model together with its
preprocessing encoders to a single file (model_artifacts.pkl).

The Streamlit app (app.py) loads that file to make predictions,
so this script must be run ONCE before launching the app.

Usage
-----
    python train_model.py

Requirements
------------
- "Mwanza_maths_updated.csv" must be in the same folder as this
  script, with the columns:
      teacher_to_student_ratio, school_type, mock_result, NECTA_result
- Install dependencies first:
      pip install -r requirements.txt
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------------------------------------------
# Configuration (must match the original notebook)
# --------------------------------------------------------------
DATA_FILE = "Mwanza_maths_updated.csv"
MODEL_FILE = "model_artifacts.pkl"

FEATURE_COLS = ["teacher_to_student_ratio", "school_type_encoded", "mock_result_encoded"]
MOCK_ORDER = [["F", "D", "C", "B", "A"]]  # worst -> best


def main():
    # ------------------------------------------------------------
    # 1. Load and clean the data
    # ------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Shape before cleaning: {df.shape}")

    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # ------------------------------------------------------------
    # 2. Encode categorical features (same scheme as the notebook)
    # ------------------------------------------------------------
    # school_type -> Government = 0, Private = 1 (alphabetical)
    le_school = LabelEncoder()
    df["school_type_encoded"] = le_school.fit_transform(df["school_type"])

    # mock_result -> ordinal F < D < C < B < A
    oe_mock = OrdinalEncoder(
        categories=MOCK_ORDER, handle_unknown="use_encoded_value", unknown_value=-1
    )
    df["mock_result_encoded"] = oe_mock.fit_transform(df[["mock_result"]]).astype(int)

    print("\nSchool type encoding:")
    for label, code in zip(le_school.classes_, le_school.transform(le_school.classes_)):
        print(f"  {label} -> {code}")

    print("\nMock result encoding (F is lowest, A is highest):")
    for grade, code in zip(MOCK_ORDER[0], range(len(MOCK_ORDER[0]))):
        print(f"  {grade} -> {code}")

    # ------------------------------------------------------------
    # 3. Build features / target and split
    # ------------------------------------------------------------
    X = df[FEATURE_COLS]
    y = df["NECTA_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # 4. Train Logistic Regression
    # ------------------------------------------------------------
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    print("\n=== Logistic Regression Results ===")
    print(f"Accuracy: {lr_acc * 100:.2f}%")
    print(classification_report(y_test, lr_pred, target_names=["Fail", "Pass"]))

    # ------------------------------------------------------------
    # 5. Train Random Forest
    # ------------------------------------------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("\n=== Random Forest Results ===")
    print(f"Accuracy: {rf_acc * 100:.2f}%")
    print(classification_report(y_test, rf_pred, target_names=["Fail", "Pass"]))

    # ------------------------------------------------------------
    # 6. Pick the best model (same rule as the notebook)
    # ------------------------------------------------------------
    if rf_acc >= lr_acc:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_acc = rf_acc
    else:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_acc = lr_acc

    print(f"\nBest model: {best_model_name} ({best_acc * 100:.2f}% accuracy)")

    # ------------------------------------------------------------
    # 7. Save everything the Streamlit app needs
    # ------------------------------------------------------------
    artifacts = {
        "model": best_model,
        "model_name": best_model_name,
        "le_school": le_school,
        "oe_mock": oe_mock,
        "feature_cols": FEATURE_COLS,
        "mock_order": MOCK_ORDER[0],
        "accuracy": best_acc,
        "lr_accuracy": lr_acc,
        "rf_accuracy": rf_acc,
    }

    joblib.dump(artifacts, MODEL_FILE)
    print(f"\nSaved model artifacts to '{MODEL_FILE}'.")
    print("You can now run the Streamlit app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
