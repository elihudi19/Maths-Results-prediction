import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE    = "Mwanza_maths_updated.csv"
MODEL_FILE   = "model.joblib"
MOCK_ORDER   = ["F", "D", "C", "B", "A"]
SCHOOL_MAP   = {"Private": 0, "Government": 1}
FEATURE_COLS = ["teacher_to_student_ratio", "school_type_encoded", "mock_result_encoded"]


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Shape before cleaning: {df.shape}")
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    df["school_type_encoded"] = df["school_type"].map(SCHOOL_MAP)

    oe_mock = OrdinalEncoder(
        categories=[MOCK_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df["mock_result_encoded"] = oe_mock.fit_transform(df[["mock_result"]]).astype(int)

    print("\nSchool type encoding (Private = 0 baseline, Government = 1):")
    for k, v in SCHOOL_MAP.items():
        print(f"  {k} -> {v}")

    print("\nMock result encoding (F is lowest, A is highest):")
    for grade, code in zip(MOCK_ORDER, range(len(MOCK_ORDER))):
        print(f"  {grade} -> {code}")

    X = df[FEATURE_COLS]
    y = df["NECTA_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    print(f"\n=== Logistic Regression Accuracy: {lr_acc * 100:.2f}% ===")
    print(classification_report(y_test, lr_model.predict(X_test), target_names=["Fail", "Pass"]))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"\n=== Random Forest Accuracy: {rf_acc * 100:.2f}% ===")
    print(classification_report(y_test, rf_model.predict(X_test), target_names=["Fail", "Pass"]))

    if rf_acc >= lr_acc:
        best_model, best_name, best_acc = rf_model, "Random Forest", rf_acc
    else:
        best_model, best_name, best_acc = lr_model, "Logistic Regression", lr_acc

    print(f"\nBest model: {best_name} ({best_acc * 100:.2f}% accuracy)")

    artifacts = {
        "model":        best_model,
        "model_name":   best_name,
        "school_map":   SCHOOL_MAP,
        "oe_mock":      oe_mock,
        "feature_cols": FEATURE_COLS,
        "mock_order":   MOCK_ORDER,
        "accuracy":     best_acc,
        "lr_accuracy":  lr_acc,
        "rf_accuracy":  rf_acc,
    }
    joblib.dump(artifacts, MODEL_FILE)
    print(f"\nSaved to '{MODEL_FILE}'. Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
