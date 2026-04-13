"""Flask REST API for student performance prediction."""
import pandas as pd
from flask import Flask, request, jsonify

from student_performance import __version__
from student_performance.data.loader import DataLoader
from student_performance.data.preprocessor import Preprocessor
from student_performance.features.engineering import FeatureEngineer
from student_performance.models.classifier import StudentClassifier
from student_performance.models.regressor import StudentRegressor
from student_performance.utils.helpers import get_logger, clip_grade

logger = get_logger(__name__)

FEATURE_KEYS = [
    "age", "gender", "study_hours", "attendance_rate",
    "previous_grade", "extracurricular", "internet_access", "parent_education",
]


def _train_models():
    """Train classifier and regressor on sample data; return fitted objects."""
    loader = DataLoader()
    df = loader.load_sample_data()

    preprocessor = Preprocessor()
    df_proc = preprocessor.fit_transform(df)

    engineer = FeatureEngineer()
    df_eng = engineer.fit_transform(df_proc)

    feature_cols = [c for c in df_eng.columns if c not in ("final_grade", "passed")]
    X = df_eng[feature_cols].values
    y_cls = df_eng["passed"].values
    y_reg = df_eng["final_grade"].values

    classifier = StudentClassifier(model_type="random_forest")
    classifier.fit(X, y_cls)

    regressor = StudentRegressor(model_type="random_forest")
    regressor.fit(X, y_reg)

    return preprocessor, engineer, classifier, regressor, feature_cols


def create_app(config=None) -> Flask:
    """Application factory."""
    app = Flask(__name__)
    if config:
        app.config.update(config)

    preprocessor, engineer, classifier, regressor, feature_cols = _train_models()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "version": __version__}), 200

    @app.route("/model/info", methods=["GET"])
    def model_info():
        return jsonify({
            "classifier": {
                "type": classifier.model_type,
                "is_fitted": classifier.is_fitted,
            },
            "regressor": {
                "type": regressor.model_type,
                "is_fitted": regressor.is_fitted,
            },
            "feature_columns": feature_cols,
        }), 200

    def _prepare_input(features: dict):
        """Convert raw feature dict to model-ready numpy array."""
        df = pd.DataFrame([features])
        df_proc = preprocessor.transform(df)
        df_eng = engineer.transform(df_proc)
        # Align columns to training feature set
        for col in feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0
        X = df_eng[feature_cols].values
        return X

    @app.route("/predict/pass", methods=["POST"])
    def predict_pass():
        body = request.get_json(silent=True)
        if not body or "features" not in body:
            return jsonify({"error": "Request body must contain 'features' key"}), 400
        features = body["features"]
        missing = [k for k in FEATURE_KEYS if k not in features]
        if missing:
            return jsonify({"error": f"Missing feature(s): {', '.join(missing)}"}), 400
        try:
            X = _prepare_input(features)
            prediction = int(classifier.predict(X)[0])
            probability = float(classifier.predict_proba(X)[0][1])
            return jsonify({"prediction": prediction, "probability": probability}), 200
        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/predict/grade", methods=["POST"])
    def predict_grade():
        body = request.get_json(silent=True)
        if not body or "features" not in body:
            return jsonify({"error": "Request body must contain 'features' key"}), 400
        features = body["features"]
        missing = [k for k in FEATURE_KEYS if k not in features]
        if missing:
            return jsonify({"error": f"Missing feature(s): {', '.join(missing)}"}), 400
        try:
            X = _prepare_input(features)
            predicted_grade = clip_grade(float(regressor.predict(X)[0]))
            return jsonify({"predicted_grade": predicted_grade}), 200
        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            return jsonify({"error": "Internal server error"}), 500

    return app
