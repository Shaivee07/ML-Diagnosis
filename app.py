"""
DiagnoAI — Flask Backend
Matches the scoring logic from disease_diagnosis_website.html
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML frontend

# ─────────────────────────────────────────────
# DATA — mirrors the JS constants in the HTML
# ─────────────────────────────────────────────

SYMPTOMS = {
    "resp":    ["Cough", "Shortness of breath", "Wheezing", "Sore throat",
                "Runny nose", "Chest tightness", "Sneezing", "Hemoptysis"],
    "cardio":  ["Chest pain", "Palpitations", "Leg swelling",
                "Dizziness", "Syncope", "Irregular heartbeat"],
    "general": ["Fever", "Fatigue", "Headache", "Night sweats", "Weight loss",
                "Chills", "Muscle aches", "Loss of appetite"],
    "other":   ["Nausea", "Vomiting", "Diarrhea", "Abdominal pain",
                "Rash", "Joint pain", "Back pain"]
}

DISEASES = [
    {"name": "Common Cold",
     "syms": ["cough","sore_throat","runny_nose","fatigue","headache","sneezing"],
     "temp": [36, 37.8], "hr": [60,100], "spo2": [95,100], "wbc": [4.5,11],
     "hist": [],
     "info": "Viral URTI. Self-limiting, usually 7–10 days."},
    {"name": "Influenza",
     "syms": ["fever","cough","muscle_aches","fatigue","headache","chills","sore_throat"],
     "temp": [38,40.5], "hr": [80,115], "spo2": [93,99], "wbc": [3,9],
     "hist": [],
     "info": "Seasonal flu. Antivirals effective if given early."},
    {"name": "COVID-19",
     "syms": ["fever","cough","shortness_of_breath","fatigue","muscle_aches","loss_of_appetite","headache"],
     "temp": [37.5,40], "hr": [70,120], "spo2": [80,97], "wbc": [3,8],
     "hist": ["diabetes","hypertension"],
     "info": "SARS-CoV-2 infection. Wide range of severity."},
    {"name": "Pneumonia",
     "syms": ["fever","cough","shortness_of_breath","chest_pain","fatigue","chills","muscle_aches"],
     "temp": [38.5,41], "hr": [90,130], "spo2": [82,94], "wbc": [12,25],
     "hist": ["copd","diabetes"],
     "info": "Lung parenchyma infection. Requires antibiotics."},
    {"name": "Asthma Attack",
     "syms": ["wheezing","shortness_of_breath","chest_tightness","cough"],
     "temp": [36,37.5], "hr": [90,130], "spo2": [82,94], "wbc": [4,12],
     "hist": ["asthma"],
     "info": "Bronchospasm episode. Bronchodilators indicated."},
    {"name": "COPD Exacerbation",
     "syms": ["shortness_of_breath","wheezing","cough","chest_tightness","fatigue"],
     "temp": [37,39], "hr": [85,130], "spo2": [78,92], "wbc": [8,18],
     "hist": ["copd"],
     "info": "Acute worsening of COPD. Typically in smokers 50+."},
    {"name": "Myocardial Infarction",
     "syms": ["chest_pain","shortness_of_breath","nausea","dizziness","palpitations","fatigue"],
     "temp": [36.5,37.5], "hr": [50,130], "spo2": [86,96], "wbc": [10,20],
     "hist": ["heart_disease","hypertension","diabetes"],
     "info": "Coronary artery occlusion. Medical emergency."},
    {"name": "Pulmonary Embolism",
     "syms": ["chest_pain","shortness_of_breath","palpitations","dizziness","leg_swelling"],
     "temp": [37,38.5], "hr": [95,140], "spo2": [80,94], "wbc": [8,18],
     "hist": ["heart_disease"],
     "info": "PE — blood clot in pulmonary vasculature."},
    {"name": "Hypertensive Crisis",
     "syms": ["headache","dizziness","chest_pain","shortness_of_breath","palpitations","nausea"],
     "temp": [36.5,37.5], "hr": [80,130], "spo2": [93,99], "wbc": [5,11],
     "hist": ["hypertension"],
     "info": "Severely elevated BP. Urgent antihypertensive treatment."},
    {"name": "Heart Failure",
     "syms": ["shortness_of_breath","leg_swelling","fatigue","palpitations","chest_tightness","dizziness"],
     "temp": [36,37.5], "hr": [80,130], "spo2": [82,94], "wbc": [5,13],
     "hist": ["heart_disease","hypertension"],
     "info": "Reduced cardiac output. Diuretics and ACE inhibitors."},
    {"name": "Gastroenteritis",
     "syms": ["nausea","vomiting","diarrhea","abdominal_pain","fever","fatigue"],
     "temp": [37.5,39.5], "hr": [75,110], "spo2": [96,100], "wbc": [5,13],
     "hist": [],
     "info": "GI infection — viral or bacterial. Supportive care."},
    {"name": "GERD",
     "syms": ["chest_pain","cough","nausea","abdominal_pain"],
     "temp": [36,37.3], "hr": [60,90], "spo2": [96,100], "wbc": [4.5,10],
     "hist": ["diabetes"],
     "info": "Acid reflux. Managed with PPIs and lifestyle changes."},
    {"name": "Allergic Rhinitis",
     "syms": ["sneezing","runny_nose","cough","headache","fatigue","wheezing"],
     "temp": [36,37.2], "hr": [60,90], "spo2": [96,100], "wbc": [4.5,11],
     "hist": ["asthma"],
     "info": "Allergic inflammation of nasal passages."},
    {"name": "Tuberculosis",
     "syms": ["cough","night_sweats","weight_loss","fever","fatigue","chest_pain"],
     "temp": [37.5,39.5], "hr": [80,110], "spo2": [88,96], "wbc": [5,15],
     "hist": ["diabetes"],
     "info": "Mycobacterium tuberculosis infection. Notifiable disease."},
    {"name": "Dengue Fever",
     "syms": ["fever","headache","muscle_aches","joint_pain","rash","nausea","fatigue"],
     "temp": [38.5,41], "hr": [60,100], "spo2": [94,99], "wbc": [1.5,5],
     "hist": [],
     "info": "Arboviral disease. Watch for hemorrhagic features."},
    {"name": "Appendicitis",
     "syms": ["abdominal_pain","nausea","vomiting","fever","loss_of_appetite"],
     "temp": [37.5,39.5], "hr": [80,120], "spo2": [96,100], "wbc": [11,20],
     "hist": [],
     "info": "Appendix inflammation. Surgical emergency if perforated."},
]

# In-memory session log (replace with a database in production)
analysis_log = []


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

def in_range(value, rng):
    """Check if a numeric value falls within [low, high]."""
    return rng[0] <= value <= rng[1]


def score_disease(disease, selected_symptoms, history, labs):
    """
    Scoring weights (mirrors the JavaScript frontend):
      - Symptom coverage  : 60 pts
      - Symptom precision : 15 pts
      - Medical history   : 10 pts  (−12 if history contradicts)
      - Lab correlation   : 15 pts
    """
    selected = set(selected_symptoms)
    matched = [s for s in disease["syms"] if s in selected]
    if not matched:
        return 0

    score = 0

    # Symptom coverage (60 %)
    score += (len(matched) / len(disease["syms"])) * 60

    # Precision bonus (15 %)
    score += (len(matched) / max(len(selected_symptoms), 1)) * 15

    # History modifier (±)
    if not disease["hist"] or history in disease["hist"]:
        score += 10
    else:
        score -= 12

    # Special boosts for history-specific diseases
    if disease["name"] == "Asthma Attack" and history == "asthma":
        score += 18
    if disease["name"] == "COPD Exacerbation" and history == "copd":
        score += 18
    if disease["name"] == "Hypertensive Crisis" and history == "hypertension":
        score += 15

    # Lab correlation (15 %)
    lab_hits, lab_total = 0, 0
    for key, rng_key in [("temp","temp"), ("hr","hr"), ("spo2","spo2"), ("wbc","wbc")]:
        val = labs.get(key)
        if val is not None:
            lab_total += 1
            if in_range(val, disease[rng_key]):
                lab_hits += 1
    if lab_total > 0:
        score += (lab_hits / lab_total) * 15

    return max(0, score)


def run_analysis(age, sex, history, labs, selected_symptoms, model):
    """Run scoring and return ranked differential diagnoses."""
    scored = []
    for d in DISEASES:
        s = score_disease(d, selected_symptoms, history, labs)
        if s > 4:
            scored.append({**d, "score": s})

    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return []

    total = sum(d["score"] for d in scored) or 1
    results = []
    for i, d in enumerate(scored[:6]):
        prob = round((d["score"] / total) * 100)
        confidence = "high" if prob >= 40 else ("moderate" if prob >= 20 else "low")
        results.append({
            "rank":       i + 1,
            "name":       d["name"],
            "info":       d["info"],
            "probability": prob,
            "confidence": confidence,
            "score":      round(d["score"], 2),
        })
    return results


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    """Return all symptom categories and their items."""
    return jsonify({"symptoms": SYMPTOMS})


@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    """Return the list of all disease names."""
    return jsonify({"diseases": [d["name"] for d in DISEASES]})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run differential diagnosis.

    Expected JSON body:
    {
        "age":      40,
        "sex":      "M",
        "history":  "none",
        "model":    "rf",
        "labs": {
            "temp": 37.0,
            "hr":   72,
            "spo2": 98,
            "wbc":  7.5
        },
        "symptoms": ["cough", "fever", "fatigue"]
    }
    """
    data = request.get_json(force=True)

    # ── Validate required fields ──
    errors = []
    symptoms = data.get("symptoms", [])
    if not symptoms:
        errors.append("At least one symptom must be selected.")

    age = data.get("age", 40)
    if not isinstance(age, (int, float)) or not (1 <= age <= 120):
        errors.append("Age must be between 1 and 120.")

    sex = data.get("sex", "M")
    if sex not in ("M", "F"):
        errors.append("Sex must be 'M' or 'F'.")

    history = data.get("history", "none")
    valid_histories = {"none","diabetes","hypertension","asthma","heart_disease","copd"}
    if history not in valid_histories:
        errors.append(f"Invalid history value: {history}")

    model = data.get("model", "rf")
    if model not in ("rf", "dt", "nb"):
        errors.append("Model must be 'rf', 'dt', or 'nb'.")

    labs_raw = data.get("labs", {})
    labs = {}
    lab_limits = {
        "temp": (34.0, 43.0),
        "hr":   (20, 300),
        "spo2": (50, 100),
        "wbc":  (0.1, 100),
    }
    for key, (lo, hi) in lab_limits.items():
        val = labs_raw.get(key)
        if val is not None:
            try:
                val = float(val)
                if not (lo <= val <= hi):
                    errors.append(f"Lab value '{key}' = {val} is out of physiological range ({lo}–{hi}).")
                else:
                    labs[key] = val
            except (ValueError, TypeError):
                errors.append(f"Lab value '{key}' must be a number.")

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    # ── Run scoring ──
    results = run_analysis(age, sex, history, labs, symptoms, model)

    if not results:
        return jsonify({
            "success": True,
            "results": [],
            "message": "No matching conditions found. Try selecting more symptoms."
        })

    # ── Log this analysis (in-memory; swap for DB in production) ──
    analysis_log.append({
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "age": age, "sex": sex, "history": history, "model": model,
        "labs": labs, "symptoms": symptoms,
        "top_result": results[0]["name"] if results else None,
    })

    model_label = {"rf": "Random Forest", "dt": "Decision Tree", "nb": "Naive Bayes"}[model]

    return jsonify({
        "success":     True,
        "model":       model_label,
        "total_matches": len([d for d in DISEASES if score_disease(d, symptoms, history, labs) > 4]),
        "results":     results,
    })


@app.route("/api/check-labs", methods=["POST"])
def check_labs():
    """
    Return colour-coded status for each lab value.
    (Used to replicate the live lab-card highlighting from the frontend.)

    Expected JSON body: { "temp": 37.0, "hr": 72, "spo2": 98, "wbc": 7.5 }
    """
    data = request.get_json(force=True)
    thresholds = {
        "temp": {"lo": 35, "hi": 41, "warn_lo": 36.1, "warn_hi": 37.8},
        "hr":   {"lo": 30, "hi": 180, "warn_lo": 60,  "warn_hi": 100},
        "spo2": {"lo": 85, "hi": 100, "warn_lo": 95,  "warn_hi": 100},
        "wbc":  {"lo": 1,  "hi": 40,  "warn_lo": 4.5, "warn_hi": 11},
    }
    status = {}
    for key, t in thresholds.items():
        val = data.get(key)
        if val is None:
            status[key] = "normal"
            continue
        val = float(val)
        if val < t["lo"] or val > t["hi"]:
            status[key] = "abnormal"
        elif val < t["warn_lo"] or val > t["warn_hi"]:
            status[key] = "warning"
        else:
            status[key] = "normal"
    return jsonify({"status": status})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the last 20 analyses (admin / debug endpoint)."""
    return jsonify({"analyses": analysis_log[-20:]})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
