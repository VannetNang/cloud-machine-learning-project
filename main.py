import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# ===== 1. Load data =====
data_path = Path("diabetes_data.csv")  # adjust path if needed
df = pd.read_csv(data_path)

# ===== 2. Basic cleaning =====
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# ===== 3. Remove outliers using IQR =====
numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

mask = np.ones(len(df), dtype=bool)
for col in numeric_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    mask &= (df[col] >= lower) & (df[col] <= upper)

df_clean = df[mask].reset_index(drop=True)

# ===== 4. Features / target =====
X = df_clean.drop(columns=["diabetes"])
y = df_clean["diabetes"]

categorical_features = ["gender", "smoking_history"]
numeric_features = [c for c in X.columns if c not in categorical_features]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# ===== 5. Train/test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 6. Define and evaluate models =====
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

results = []
best_model_name = None
best_f1 = -1
best_pipe = None

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_score": f1,
        }
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipe = pipe

results_df = pd.DataFrame(results).sort_values(by="F1_score", ascending=False)
print(results_df)
print("Best model:", best_model_name)

# ===== 7. Save best model as pickle =====
with open("diabetes_best_model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

print("Model saved to diabetes_best_model.pkl")

# ===== 8. Example predictions for the two persons =====
persons = pd.DataFrame(
    [
        {
            "gender": "Female",
            "age": 50,
            "hypertension": 1,
            "heart_disease": 1,
            "smoking_history": "No Info",
            "bmi": 32.35,
            "HbA1c_level": 6.23,
            "blood_glucose_level": 80,
        },
        {
            "gender": "Male",
            "age": 30,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "current",
            "bmi": 26.15,
            "HbA1c_level": 4.23,
            "blood_glucose_level": 90,
        },
    ]
)

preds = best_pipe.predict(persons)
probs = best_pipe.predict_proba(persons)[:, 1]

for i, (p, prob) in enumerate(zip(preds, probs), start=1):
    label = "Diabetic" if p == 1 else "Non-diabetic"
    print(
        f"Person {i}: prediction = {p} ({label}), probability of diabetes = {prob:.6f}"
    )
