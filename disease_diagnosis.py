"""
Disease Diagnosis System
========================
Uses YOUR OWN dataset (disease_dataset.csv) instead of generating synthetic data.

Run:
  pip install scikit-learn pandas numpy
  python disease_diagnosis.py
  
Make sure disease_dataset.csv is in the same folder as this file.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG — change this path if needed
# ─────────────────────────────────────────────

DATASET_PATH = "disease_dataset.csv"   # ← your CSV file

ALL_HISTORY = ["none", "diabetes", "hypertension", "asthma", "heart_disease", "copd"]

# ─────────────────────────────────────────────
#  1. LOAD YOUR DATASET
# ─────────────────────────────────────────────

def load_dataset(path):
    df = pd.read_csv(path)

    print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Diseases      : {df['diagnosis'].nunique()} classes")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # ── Rename columns to match model internals ──
    df = df.rename(columns={
        "medical_history":  "history_name",
        "temperature_c":    "temperature",
        "heart_rate_bpm":   "heart_rate",
        "spo2_pct":         "spo2",
        "wbc_thou_per_ul":  "wbc",
        "diagnosis":        "label"
    })

    # Encode sex: M=0, F=1
    df["sex"] = (df["sex"].str.upper() == "F").astype(int)

    # Encode history as integer index
    df["history"] = df["history_name"].apply(
        lambda h: ALL_HISTORY.index(h) if h in ALL_HISTORY else 0
    )

    # Drop columns not needed
    df = df.drop(columns=["patient_id", "history_name"], errors="ignore")

    print("\nClass distribution:")
    for disease, count in df["label"].value_counts().items():
        print(f"  {disease:<30} {count}")

    return df


# ─────────────────────────────────────────────
#  2. DETECT SYMPTOM COLUMNS AUTOMATICALLY
# ─────────────────────────────────────────────

def get_symptom_cols(df):
    non_symptom = {"age","sex","history","temperature","heart_rate",
                   "spo2","wbc","label","fever_flag","high_hr_flag",
                   "low_spo2_flag","high_wbc_flag","low_wbc_flag","symptom_count"}
    return [c for c in df.columns
            if c not in non_symptom
            and df[c].dropna().isin([0,1]).all()]


# ─────────────────────────────────────────────
#  3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df, symptom_cols):
    df = df.copy()
    df["fever_flag"]    = (df["temperature"] >= 38.0).astype(int)
    df["high_hr_flag"]  = (df["heart_rate"] > 100).astype(int)
    df["low_spo2_flag"] = (df["spo2"] < 94).astype(int)
    df["high_wbc_flag"] = (df["wbc"] > 11.0).astype(int)
    df["low_wbc_flag"]  = (df["wbc"] < 4.5).astype(int)
    df["symptom_count"] = df[symptom_cols].sum(axis=1)
    return df

def get_feature_cols(symptom_cols):
    base = ["age","sex","history","temperature","heart_rate","spo2","wbc",
            "fever_flag","high_hr_flag","low_spo2_flag","high_wbc_flag",
            "low_wbc_flag","symptom_count"]
    return base + symptom_cols


# ─────────────────────────────────────────────
#  4. TRAIN ALL THREE MODELS
# ─────────────────────────────────────────────

def train_models(df, symptom_cols):
    df = engineer_features(df, symptom_cols)
    feature_cols = get_feature_cols(symptom_cols)

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Features used : {len(feature_cols)}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42
        ),
        "Naive Bayes": GaussianNB()
    }

    trained = {}
    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        bar   = "█" * int(acc * 40)
        print(f"\n  {name:<20}  {acc*100:.1f}%  {bar}")
        trained[name] = clf

    # Detailed report
    best      = trained["Random Forest"]
    best_pred = best.predict(X_test)
    print("\n" + "="*55)
    print("  DETAILED REPORT — Random Forest")
    print("="*55)
    print(classification_report(y_test, best_pred, target_names=le.classes_))

    # Feature importances
    fi = sorted(zip(feature_cols, best.feature_importances_),
                key=lambda x: x[1], reverse=True)
    print("Top 10 important features:")
    for feat, imp in fi[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    return trained, le, feature_cols


# ─────────────────────────────────────────────
#  5. PREDICTION ENGINE
# ─────────────────────────────────────────────

def predict_diagnosis(model, le, feature_cols, symptom_cols,
                      age, sex, history, symptoms,
                      temperature, heart_rate, spo2, wbc, top_k=5):
    row = {col: 0 for col in feature_cols}
    row["age"]         = float(age)
    row["sex"]         = 1 if str(sex).upper() == "F" else 0
    row["history"]     = ALL_HISTORY.index(history) if history in ALL_HISTORY else 0
    row["temperature"] = float(temperature)
    row["heart_rate"]  = int(heart_rate)
    row["spo2"]        = int(spo2)
    row["wbc"]         = float(wbc)

    for sym in symptoms:
        key = sym.lower().replace(" ","_")
        if key in row:
            row[key] = 1

    row["fever_flag"]    = int(float(temperature) >= 38.0)
    row["high_hr_flag"]  = int(int(heart_rate) > 100)
    row["low_spo2_flag"] = int(int(spo2) < 94)
    row["high_wbc_flag"] = int(float(wbc) > 11.0)
    row["low_wbc_flag"]  = int(float(wbc) < 4.5)
    row["symptom_count"] = sum(row[s] for s in symptom_cols if s in row)

    X = pd.DataFrame([row])[feature_cols]

    if hasattr(model, "predict_proba"):
        probs      = model.predict_proba(X)[0]
        ranked_idx = np.argsort(probs)[::-1][:top_k]
        return [(le.classes_[i], round(float(probs[i])*100, 1))
                for i in ranked_idx if probs[i] > 0.01]
    else:
        idx = int(model.predict(X)[0])
        return [(le.classes_[idx], 100.0)]


# ─────────────────────────────────────────────
#  6. PRETTY PRINTER
# ─────────────────────────────────────────────

def print_results(results, patient_info):
    print("\n" + "="*55)
    print("  DIAGNOSIS RESULTS")
    print("="*55)
    print(f"  Patient : Age {patient_info['age']}, {patient_info['sex']}")
    print(f"  History : {patient_info['history']}")
    print(f"  Symptoms: {', '.join(patient_info['symptoms']) or 'none'}")
    print(f"  Labs    : Temp={patient_info['temp']}°C  HR={patient_info['hr']}bpm  "
          f"SpO₂={patient_info['spo2']}%  WBC={patient_info['wbc']}")
    print("-"*55)
    medals = ["🥇","🥈","🥉"]
    for i, (disease, prob) in enumerate(results):
        tag   = " ← TOP MATCH" if i == 0 else ""
        medal = medals[i] if i < 3 else f"  {i+1}."
        bar   = "▓" * int(prob/3) + "░" * max(0, 33-int(prob/3))
        print(f"  {medal}  {disease:<28} {prob:5.1f}%  {bar}{tag}")
    print("="*55)
    print("  ⚠  For educational use only. Consult a doctor.\n")


# ─────────────────────────────────────────────
#  7. DEMO
# ─────────────────────────────────────────────

def run_demo(trained, le, feature_cols, symptom_cols):
    demo_cases = [
        {"label":"Classic flu","age":28,"sex":"M","history":"none",
         "symptoms":["fever","cough","muscle_aches","fatigue","chills","headache"],
         "temperature":39.2,"heart_rate":95,"spo2":97,"wbc":5.5},
        {"label":"Heart attack","age":58,"sex":"M","history":"hypertension",
         "symptoms":["chest_pain","shortness_of_breath","nausea","dizziness","palpitations"],
         "temperature":37.1,"heart_rate":112,"spo2":91,"wbc":14.2},
        {"label":"Asthma patient","age":22,"sex":"F","history":"asthma",
         "symptoms":["wheezing","shortness_of_breath","chest_pain","cough"],
         "temperature":37.0,"heart_rate":105,"spo2":88,"wbc":8.1},
    ]
    model = trained["Random Forest"]
    print("\n" + "="*55)
    print("  DEMO — 3 TEST CASES")
    print("="*55)
    for case in demo_cases:
        print(f"\n[CASE] {case['label']}")
        results = predict_diagnosis(model, le, feature_cols, symptom_cols,
                                    case["age"],case["sex"],case["history"],
                                    case["symptoms"],case["temperature"],
                                    case["heart_rate"],case["spo2"],case["wbc"])
        print_results(results,{"age":case["age"],"sex":case["sex"],
                               "history":case["history"],"symptoms":case["symptoms"],
                               "temp":case["temperature"],"hr":case["heart_rate"],
                               "spo2":case["spo2"],"wbc":case["wbc"]})


# ─────────────────────────────────────────────
#  8. INTERACTIVE
# ─────────────────────────────────────────────

def interactive_mode(trained, le, feature_cols, symptom_cols):
    print("\n" + "="*55)
    print("  INTERACTIVE MODE")
    print("="*55)
    while True:
        print("─"*55)
        age_in = input("Age (or 'quit'): ").strip()
        if age_in.lower() in ["quit","exit","q"]: break

        model_choice = input("Model (RF/DT/NB) [RF]: ").strip().upper() or "RF"
        model = trained.get(
            {"RF":"Random Forest","DT":"Decision Tree","NB":"Naive Bayes"}.get(model_choice,"Random Forest"),
            trained["Random Forest"])

        sex     = input("Sex (M/F) [M]: ").strip() or "M"
        print(f"History: {', '.join(ALL_HISTORY)}")
        history = input("Medical history [none]: ").strip().lower() or "none"
        if history not in ALL_HISTORY: history = "none"

        print("\nSymptoms:")
        for i,s in enumerate(symptom_cols,1): print(f"  {i:2d}. {s}")
        syms_in  = input("\nEnter symptoms (names or numbers): ").strip()
        symptoms = []
        for part in syms_in.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)-1
                if 0 <= idx < len(symptom_cols): symptoms.append(symptom_cols[idx])
            else:
                key = part.lower().replace(" ","_")
                if key in symptom_cols: symptoms.append(key)

        temp = float(input("Temperature °C [37.0]: ").strip() or 37.0)
        hr   = int(input("Heart rate bpm [72]: ").strip() or 72)
        spo2 = int(input("SpO₂ % [98]: ").strip() or 98)
        wbc  = float(input("WBC ×10³/µL [7.5]: ").strip() or 7.5)

        results = predict_diagnosis(model, le, feature_cols, symptom_cols,
                                    age_in, sex, history, symptoms, temp, hr, spo2, wbc)
        print_results(results,{"age":age_in,"sex":sex,"history":history,
                               "symptoms":symptoms,"temp":temp,"hr":hr,
                               "spo2":spo2,"wbc":wbc})
        if input("Another? (y/N): ").strip().lower() != "y": break
    print("Goodbye!")


# ─────────────────────────────────────────────
#  9. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  DISEASE DIAGNOSIS SYSTEM")
    print(f"  Dataset: {DATASET_PATH}")
    print("="*55)

    df           = load_dataset(DATASET_PATH)
    symptom_cols = get_symptom_cols(df)

    print(f"\nSymptoms found: {len(symptom_cols)}")

    trained, le, feature_cols = train_models(df, symptom_cols)
    run_demo(trained, le, feature_cols, symptom_cols)

    if input("\nLaunch interactive mode? (y/N): ").strip().lower() == "y":
        interactive_mode(trained, le, feature_cols, symptom_cols)
