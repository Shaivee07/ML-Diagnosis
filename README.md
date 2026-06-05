🧠 ML-Diagnosis (DiagnoAI)

ML-based Differential Diagnosis System using Python & Flask

An end-to-end machine learning powered disease diagnosis system that predicts possible medical conditions based on symptoms, vital signs, and patient history.

🚀 Project Overview

ML-Diagnosis (DiagnoAI) is a full-stack application that combines classical machine learning models with a responsive web interface to provide real-time disease predictions.

Predicts 15 different diseases
Uses symptoms + vitals + medical history
Returns top-ranked diagnoses with probability scores
Works via CLI tool + Web App
🎯 Key Features

✅ Multi-model prediction (Random Forest, Decision Tree, Naïve Bayes)
✅ Top-5 ranked disease predictions with confidence %
✅ 25 symptom inputs + vital signs support
✅ Interactive web UI (no framework needed)
✅ REST API with JSON responses
✅ CLI mode for batch predictions
✅ Real-time scoring (frontend + backend)

🧬 Supported Diseases
Respiratory: Cold, Influenza, COVID-19, Pneumonia, Asthma, COPD, Tuberculosis
Cardiovascular: Heart Attack, Pulmonary Embolism, Hypertensive Crisis
Gastrointestinal: GERD, Appendicitis
Infectious/Systemic: Dengue, Gastroenteritis
Allergic conditions

📊 Dataset
📁 3,000 patient records
🧾 34 features:
25 symptoms (binary)
4 vital signs (Temp, HR, SpO₂, WBC)
Age, Sex, Medical History
🎯 15 disease classes
⚙️ ML Pipeline
Data preprocessing & encoding
Feature engineering (vital flags, symptom count)
Train-test split (80/20)
Model training (RF, DT, NB)
Prediction using probability ranking
📈 Model Performance
Model	Accuracy	Role
Random Forest	~92%	Primary Model
Decision Tree	~84%	Interpretable Model
Naïve Bayes	~80%	Fast Baseline
