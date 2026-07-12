from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_decision_tree.pkl"

FEATURE_COLUMNS = [
    "Age",
    "FrequentFlyer",
    "AnnualIncomeClass",
    "ServicesOpted",
    "AccountSyncedToSocialMedia",
    "BookedHotelOrNot",
]
TARGET_COLUMN = "Target"

TARGET_LABELS = {
    0: "Baixo risco de churn",
    1: "Alto risco de churn",
}

INCOME_CLASS_MAP = {
    "low income": 0,
    "middle income": 1,
    "high income": 2,
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2,
}

YES_NO_MAP = {
    "no": 0,
    "yes": 1,
    "false": 0,
    "true": 1,
    "0": 0,
    "1": 1,
    False: 0,
    True: 1,
    0: 0,
    1: 1,
}


class InputValidationError(ValueError):
    """Raised when a prediction payload has missing or invalid fields."""

    def __init__(self, errors: Mapping[str, str]) -> None:
        self.errors = dict(errors)
        message = "; ".join(f"{field}: {error}" for field, error in self.errors.items())
        super().__init__(message)


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the web interface is started before model training."""


def _payload_value(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _parse_int(
    value: Any,
    field: str,
    errors: dict[str, str],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None or value == "":
        errors[field] = "Campo obrigatorio."
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors[field] = "Use um numero inteiro."
        return None

    if minimum is not None and parsed < minimum:
        errors[field] = f"Use um valor maior ou igual a {minimum}."
        return None
    if maximum is not None and parsed > maximum:
        errors[field] = f"Use um valor menor ou igual a {maximum}."
        return None
    return parsed


def _map_category(
    value: Any,
    field: str,
    mapping: Mapping[Any, int],
    errors: dict[str, str],
) -> int | None:
    if value is None or value == "":
        errors[field] = "Campo obrigatorio."
        return None

    normalized = value.strip().lower() if isinstance(value, str) else value
    if normalized not in mapping:
        errors[field] = "Opcao invalida."
        return None
    return mapping[normalized]


def encode_prediction_payload(payload: Mapping[str, Any]) -> tuple[list[int], dict[str, int]]:
    """Validate and convert web form values to the feature order used in training."""

    errors: dict[str, str] = {}
    encoded = {
        "Age": _parse_int(_payload_value(payload, "Age", "age"), "Age", errors, 18, 100),
        "FrequentFlyer": _map_category(
            _payload_value(payload, "FrequentFlyer", "frequentFlyer"),
            "FrequentFlyer",
            YES_NO_MAP,
            errors,
        ),
        "AnnualIncomeClass": _map_category(
            _payload_value(payload, "AnnualIncomeClass", "annualIncomeClass"),
            "AnnualIncomeClass",
            INCOME_CLASS_MAP,
            errors,
        ),
        "ServicesOpted": _parse_int(
            _payload_value(payload, "ServicesOpted", "servicesOpted"),
            "ServicesOpted",
            errors,
            1,
            6,
        ),
        "AccountSyncedToSocialMedia": _map_category(
            _payload_value(payload, "AccountSyncedToSocialMedia", "accountSyncedToSocialMedia"),
            "AccountSyncedToSocialMedia",
            YES_NO_MAP,
            errors,
        ),
        "BookedHotelOrNot": _map_category(
            _payload_value(payload, "BookedHotelOrNot", "bookedHotelOrNot"),
            "BookedHotelOrNot",
            YES_NO_MAP,
            errors,
        ),
    }

    if errors:
        raise InputValidationError(errors)

    typed_encoded = {field: int(value) for field, value in encoded.items()}
    features = [typed_encoded[field] for field in FEATURE_COLUMNS]
    return features, typed_encoded


def _relative_or_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_model_artifact(
    data_path: str | Path = DEFAULT_CLEAN_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Train the deployment model and persist a pickle artifact plus JSON metadata."""

    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Instale as dependencias com `pip install -r requirements.txt` antes de treinar."
        ) from exc

    data_path = Path(data_path)
    model_path = Path(model_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset limpo nao encontrado em {data_path}. "
            "Execute clean-data.ipynb para gerar data/cleaned_dataset.csv."
        )

    df = pd.read_csv(data_path)
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no dataset limpo: {', '.join(missing_columns)}")

    df = df[required_columns].dropna()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(max_depth=7, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    metadata = {
        "model_name": "Decision Tree (depth=7)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data": _relative_or_string(data_path),
        "row_count": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_labels": TARGET_LABELS,
        "metrics": metrics,
    }
    artifact = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelArtifactNotFoundError(
            f"Modelo nao encontrado em {model_path}. Rode `python scripts/train_bonus_model.py`."
        )

    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if "model" not in artifact or "scaler" not in artifact:
        raise ValueError("Artefato de modelo invalido: chaves esperadas nao encontradas.")
    return artifact


def _as_model_input(features: list[int], feature_columns: list[str]) -> Any:
    try:
        import pandas as pd
    except ImportError:
        return [features]
    return pd.DataFrame([features], columns=feature_columns)


def predict_payload(
    payload: Mapping[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    features, encoded_features = encode_prediction_payload(payload)
    artifact = load_model_artifact(model_path)
    metadata = artifact.get("metadata", {})
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
    model_input = _as_model_input(features, feature_columns)

    scaled_input = artifact["scaler"].transform(model_input)
    model = artifact["model"]
    prediction = int(model.predict(scaled_input)[0])

    probability_churn = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        classes = [int(value) for value in model.classes_]
        if 1 in classes:
            probability_churn = float(probabilities[classes.index(1)])

    return {
        "prediction": prediction,
        "label": TARGET_LABELS.get(prediction, str(prediction)),
        "probability_churn": probability_churn,
        "encoded_features": encoded_features,
        "model_name": metadata.get("model_name", "Decision Tree (depth=7)"),
        "metrics": metadata.get("metrics", {}),
    }
