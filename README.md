# Student Performance Analysis and Prediction System

An MSc-level machine learning project that analyses student academic data and predicts both pass/fail outcomes and final grade scores using a full ML pipeline exposed through a REST API.

---

## Features

- **Data Generation & Loading** – synthetic 500-row dataset or custom CSV ingestion  
- **Preprocessing** – label encoding, median imputation, column normalisation  
- **Feature Engineering** – interaction terms, grade improvement, high-performer flag  
- **Classification** – Random Forest / Logistic Regression / SVM (pass/fail)  
- **Regression** – Random Forest / Linear Regression / Ridge (final grade)  
- **Model Evaluation** – accuracy, F1, ROC-AUC, RMSE, R², cross-validation  
- **REST API** – Flask application with prediction and health endpoints  
- **Test Suite** – pytest coverage across all modules  

---

## Tech Stack

| Layer | Library |
|-------|---------|
| ML | scikit-learn |
| Data | pandas, numpy |
| API | Flask |
| Serialisation | joblib |
| Visualisation | matplotlib, seaborn |
| Testing | pytest, pytest-cov |

---

## Installation

```bash
git clone <repo-url>
cd Mproj

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Install dev dependencies (for testing)
pip install -r requirements-dev.txt
```

---

## Usage

### Run the API server

```python
from student_performance.api.app import create_app

app = create_app()
app.run(debug=True, port=5000)
```

### Programmatic usage

```python
from student_performance.data.loader import DataLoader
from student_performance.data.preprocessor import Preprocessor
from student_performance.features.engineering import FeatureEngineer
from student_performance.models.classifier import StudentClassifier
from student_performance.models.evaluator import Evaluator

loader = DataLoader()
df = loader.load_sample_data()

prep = Preprocessor()
df_proc = prep.fit_transform(df)

fe = FeatureEngineer()
df_eng = fe.fit_transform(df_proc)

feature_cols = [c for c in df_eng.columns if c not in ("final_grade", "passed")]
X = df_eng[feature_cols].values
y = df_eng["passed"].values

clf = StudentClassifier(model_type="random_forest")
clf.fit(X, y)

evaluator = Evaluator()
report = evaluator.classification_report(y, clf.predict(X), clf.predict_proba(X))
print(report)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict/pass` | Predict pass (1) or fail (0) |
| POST | `/predict/grade` | Predict final grade (0–100) |

### Example: Predict pass/fail

```bash
curl -X POST http://localhost:5000/predict/pass \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 21,
      "gender": "F",
      "study_hours": 12,
      "attendance_rate": 0.9,
      "previous_grade": 70,
      "extracurricular": 1,
      "internet_access": 1,
      "parent_education": "tertiary"
    }
  }'
```

Response:
```json
{"prediction": 1, "probability": 0.87}
```

---

## Project Structure

```
├── src/student_performance/
│   ├── data/           # loader.py, preprocessor.py
│   ├── features/       # engineering.py
│   ├── models/         # classifier.py, regressor.py, evaluator.py
│   ├── api/            # app.py (Flask)
│   └── utils/          # helpers.py
├── tests/              # pytest test suite
├── requirements.txt
├── requirements-dev.txt
└── setup.py
```

---

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=student_performance --cov-report=term-missing
```
