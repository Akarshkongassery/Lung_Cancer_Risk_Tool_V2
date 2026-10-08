"""
GP-facing symptomatic lung-cancer risk assessment demonstrator.

Run:
    pip install streamlit
    streamlit run lung_cancer_gp_tool.py

The bundled scoring function is a deterministic UI placeholder only. It is not
a trained or clinically validated model. Replace PlaceholderRiskModel in the
model registry with the exported MIMIC, eICU, CPRD Aurum and SynPre-FL
pipelines when they are ready.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent
import math
import pandas as pd
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import streamlit as st
import torch
from torch import nn


APP_TITLE = "PHASE IV AI Lung Cancer Risk Assessment"
APP_SHORT_TITLE = "Lung Cancer Use-Case Demonstrator"
APP_VERSION = "PHASE IV AI prototype 2.1"
# Experimental sensitivity-analysis factor for CPRD standalone.
# This is manually selected and is not a validated recalibration.
EXPERIMENTAL_CPRD_FACTOR = 3.5
PROJECT_URL = "https://www.phase4ai-project.eu/"
PROJECT_NUMBER = "101095384"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    population: str
    threshold: float
    colour: str
    description: str
    research_only: bool = False
    


MODEL_SPECS: Dict[str, ModelSpec] = {
    "cprd": ModelSpec(
        model_id="cprd",
        display_name="Federated model from CPRD Aurum",
        population="CPRD Aurum primary-care cohort",
        threshold=0.030,  # Temporary UI threshold
        colour="#007C83",
        description=(
            "MLP trained on federated schema using the "
            "CPRD Aurum cohort."
        ),
    ),
    "cprd_varha": ModelSpec(
        model_id="cprd_varha",
        display_name="CPRD + VARHA federated model",
        population="Federated CPRD Aurum and VARHA cohorts",
        threshold=0.030,  # Temporary UI threshold
        colour="#3156A3",
        description=(
    "Federated MLP trained across harmonised CPRD Aurum "
    "and VARHA cohorts using client-specific adaptive local "
    "training and equal server aggregation. The final model "
    "was selected using mean validation AUROC."
),
    ),
        "mimic": ModelSpec(
        model_id="mimic",
        display_name=(
            "Federated Model on MIMIC-IV "
            "silver-phenotype Data"
        ),
        population=(
            "MIMIC-IV hospital-admission cohort"
        ),
        threshold=0.61,
        colour="#7B61A8",
        description=(
            " Model trained via federated scheme to detect "
            "a high-confidence derived lung-cancer silver "
            "phenotype in MIMIC-IV admissions."
        ),
        research_only=True,
    ),

    "eicu": ModelSpec(
        model_id="eicu",
        display_name=(
            "Federated Model on eICU"
            "silver-phenotype model"
        ),
        population="eICU critical-care cohort",
        threshold=0.05,
        colour="#A86732",
        description=(
            "Model trained via federated scheme to detect "
            "a high-confidence derived lung-cancer silver "
            "phenotype in eICU admissions."
        ),
        research_only=True,
    ),


}

MODEL_FEATURES: Dict[str, List[str]] = {
    "cprd": [
        "age",
        "copd_emphysema",
        "ckd",
        "cerebrovascular",
        "cardiovascular",
        "liver",
    ],
    "cprd_varha": [
        "age",
        "copd_emphysema",
        "ckd",
        "cerebrovascular",
        "cardiovascular",
        "liver",
    ],
}

FEATURE_LABELS = {
    "age": "Age",
    "gender_1": "Sex recorded as male",
    "smk_qt_final_2_2": "Current or previous smoking exposure",
    "alc_units_day_7": "Alcohol intake",
    "bmifinal2": "BMI",
    "famhlg_1": "Relevant family health history",
    "familyhcancer_1": "Family history of cancer",
    "thyroid_ca_1": "Previous thyroid cancer",
    "stomach_ca_1": "Previous stomach cancer",
    "kidney_ca_1": "Previous kidney cancer",
    "copd_com": "COPD",
    "copd_emphysema": "COPD or emphysema",
"ckd": "Chronic kidney disease",
"cerebrovascular": "Cerebrovascular disease",
"cardiovascular": ("Ischaemic heart disease or previous myocardial infarction"),
"liver": "Liver disease",
    "ovary_ca_1": "Previous ovarian cancer",
    "lek_1": "Previous leukaemia",
    "myeloma_1": "Previous myeloma",
    "melanoma_1": "Previous melanoma",
    "hdnk_ca_1": "Previous head/neck cancer",
    "bladder_ca_1": "Previous bladder cancer",
    "pancreas_1": "Previous pancreatic cancer",
    "dyspnoeaoneyear": "Breathlessness in the previous year",
    "haemoptysisoneyear": "Haemoptysis in the previous year",
    "sputumoneyear": "Sputum symptoms in the previous year",
    "weightlossoneyear": "Unexplained weight loss in the previous year",
}


@dataclass
class Prediction:
    model_id: str
    model_name: str
    probability: float
    threshold: float
    category: str
    input_coverage: float
    imputed_features: List[str]
    warnings: List[str]
    contributions: List[Tuple[str, float]]

    raw_score: Optional[float] = None
    calibration_method: Optional[str] = None
    calibration_population: Optional[str] = None
    prediction_horizon_months: Optional[int] = None
    calibrated: bool = False
    placeholder: bool = True
    original_calibrated_probability: Optional[float] = None
    experimental_adjustment_factor: Optional[float] = None
    experimentally_adjusted: bool = False
class CPRDStandaloneMLP(nn.Module):
    """Exact architecture of the standalone CPRD model."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

class CPRDStandaloneModelAdapter:
    """Inference adapter for the standalone CPRD model."""

    REQUIRED_FEATURES = [
        "gender",
        "age_at_incidence",
        "cerebrovascular_disease",
        "chronic_kidney_disease",
        "copd_emphysema",
        "ischemic_heart_myocardial_infarction",
        "peripheral_vascular",
        "bowel_cancer",
        "osteoarthritis",
        "rheumatoid_arthritis",
        "liver_disease",
        "smoking_status_N",
        "smoking_status_U",
        "smoking_status_X",
        "smoking_status_Y",
        "comorbidity_count",
        "cardio_risk",
        "any_comorbidity",
    ]

    def __init__(
    self,
    spec: ModelSpec,
    checkpoint_path: Path,
    calibrator_path: Path,
    calibration_metadata_path: Path,):
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.calibrator_path = Path(calibrator_path)
        self.calibration_metadata_path = Path(
            calibration_metadata_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Standalone CPRD checkpoint was not found at: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "The standalone CPRD checkpoint must contain "
                "a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "The standalone CPRD checkpoint does not contain "
                "'model_state_dict'."
            )

        checkpoint_features = checkpoint.get("features")

        if checkpoint_features != self.REQUIRED_FEATURES:
            raise ValueError(
                "The standalone CPRD checkpoint feature order "
                "does not match the expected feature order. "
                f"Checkpoint: {checkpoint_features}. "
                f"Expected: {self.REQUIRED_FEATURES}."
            )

        self.age_mean = float(
            checkpoint["age_scaler_mean"]
        )

        self.age_scale = float(
            checkpoint["age_scaler_scale"]
        )

        if self.age_scale <= 0:
            raise ValueError(
                "The CPRD age-scaler scale must be greater than zero."
            )

        self.model = CPRDStandaloneMLP()

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to("cpu")
        self.model.eval()

        self.metadata = {
            "features": checkpoint_features,
            "architecture": checkpoint.get("architecture"),
            "best_epoch": checkpoint.get("best_epoch"),
            "age_scaler_mean": self.age_mean,
            "age_scaler_scale": self.age_scale,
        }
        if not self.calibrator_path.exists():
            raise FileNotFoundError(
                "Standalone CPRD calibrator was not found at: "
                f"{self.calibrator_path}"
            )

        self.calibrator = joblib.load(
            self.calibrator_path
        )

        if not hasattr(
            self.calibrator,
            "predict_proba",
        ):
            raise TypeError(
                "The standalone CPRD calibrator does not "
                "provide predict_proba()."
            )

        if not self.calibration_metadata_path.exists():
            raise FileNotFoundError(
                "Standalone CPRD calibration metadata was "
                "not found at: "
                f"{self.calibration_metadata_path}"
            )

        with self.calibration_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            self.calibration_metadata = json.load(
                metadata_file
            )

        required_metadata_fields = [
            "calibration_method",
            "calibration_input",
            "calibration_population",
            "outcome_definition",
            "prediction_horizon_months",
            "calibrated_threshold",
        ]

        missing_metadata_fields = [
            field_name
            for field_name in required_metadata_fields
            if field_name
            not in self.calibration_metadata
        ]

        if missing_metadata_fields:
            raise KeyError(
                "Standalone CPRD calibration metadata is "
                "missing: "
                + ", ".join(missing_metadata_fields)
            )

        if (
            self.calibration_metadata[
                "calibration_input"
            ]
            != "raw_logit"
        ):
            raise ValueError(
                "The standalone CPRD calibrator must use "
                "raw_logit as its input."
            )

        self.threshold = float(
            self.calibration_metadata[
                "calibrated_threshold"
            ]
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "The standalone CPRD calibrated threshold "
                "must be between zero and one."
            )

        self.metadata.update(
            {
                "calibrated_threshold": self.threshold,
                "calibration_method": (
                    self.calibration_metadata[
                        "calibration_method"
                    ]
                ),
                "calibration_population": (
                    self.calibration_metadata[
                        "calibration_population"
                    ]
                ),
                "outcome_definition": (
                    self.calibration_metadata[
                        "outcome_definition"
                    ]
                ),
                "prediction_horizon_months": (
                    self.calibration_metadata[
                        "prediction_horizon_months"
                    ]
                ),
            }
        )

    @staticmethod
    def _binary_value(value: Any) -> Tuple[float, bool]:
        """Convert a UI value into zero or one."""

        if value is None:
            return 0.0, True

        if isinstance(value, str):
            normalised = value.strip().lower()

            if normalised in {"yes", "true", "1", "1.0"}:
                return 1.0, False

            if normalised in {"no", "false", "0", "0.0"}:
                return 0.0, False

            return 0.0, True

        try:
            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                return 0.0, True

            return (1.0 if numeric_value > 0 else 0.0), False

        except (TypeError, ValueError):
            return 0.0, True

    def _prepare_input(
        self,
        patient_record: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, List[str], float]:
        """
        Reproduce the preprocessing used when the standalone
        CPRD model was trained.
        """

        imputed_features: List[str] = []

        # Age
        raw_age = patient_record.get("age")

        try:
            age = float(raw_age)

            if not np.isfinite(age):
                raise ValueError

        except (TypeError, ValueError):
            age = self.age_mean
            imputed_features.append("age")

        age = float(np.clip(age, 18.0, 110.0))
        scaled_age = (age - self.age_mean) / self.age_scale

        # CPRD variables that genuinely varied during training
        source_features = {
            "cerebrovascular_disease": "cerebrovascular",
            "chronic_kidney_disease": "ckd",
            "copd_emphysema": "copd_emphysema",
            "ischemic_heart_myocardial_infarction": "cardiovascular",
            "liver_disease": "liver",
        }

        binary_values: Dict[str, float] = {}

        for model_feature, ui_feature in source_features.items():
            value, was_imputed = self._binary_value(
                patient_record.get(ui_feature)
            )

            binary_values[model_feature] = value

            if was_imputed:
                imputed_features.append(ui_feature)

        cerebrovascular = binary_values[
            "cerebrovascular_disease"
        ]
        chronic_kidney = binary_values[
            "chronic_kidney_disease"
        ]
        copd = binary_values[
            "copd_emphysema"
        ]
        ischaemic_heart = binary_values[
            "ischemic_heart_myocardial_infarction"
        ]
        liver = binary_values[
            "liver_disease"
        ]

        # These were unavailable or disabled in the CPRD
        # training pipeline and were therefore always zero.
        gender = 0.0
        peripheral_vascular = 0.0
        bowel_cancer = 0.0
        osteoarthritis = 0.0
        rheumatoid_arthritis = 0.0

        smoking_n = 0.0
        smoking_u = 0.0
        smoking_x = 0.0
        smoking_y = 0.0

        # Same derived variables used during training
        comorbidity_count = (
            cerebrovascular
            + chronic_kidney
            + copd
            + ischaemic_heart
            + peripheral_vascular
            + bowel_cancer
            + osteoarthritis
            + rheumatoid_arthritis
            + liver
        )

        cardio_risk = (
            cerebrovascular
            + ischaemic_heart
            + peripheral_vascular
        )

        any_comorbidity = (
            1.0 if comorbidity_count > 0 else 0.0
        )

        transformed_values = [
            gender,
            scaled_age,
            cerebrovascular,
            chronic_kidney,
            copd,
            ischaemic_heart,
            peripheral_vascular,
            bowel_cancer,
            osteoarthritis,
            rheumatoid_arthritis,
            liver,
            smoking_n,
            smoking_u,
            smoking_x,
            smoking_y,
            comorbidity_count,
            cardio_risk,
            any_comorbidity,
        ]

        if len(transformed_values) != 18:
            raise RuntimeError(
                "The CPRD adapter did not construct exactly "
                "18 model inputs."
            )

        tensor = torch.tensor(
            [transformed_values],
            dtype=torch.float32,
        )

        observed_count = 6 - len(
            set(imputed_features)
        )

        coverage = max(
            0.0,
            min(1.0, observed_count / 6.0),
        )

        return tensor, imputed_features, coverage

    def predict_raw(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a technical raw prediction.

        This is not yet a calibrated cancer probability and
        must not be displayed clinically.
        """

        tensor, imputed_features, coverage = (
            self._prepare_input(features)
        )

        with torch.inference_mode():
            raw_logit = float(
                self.model(tensor)[0].item()
            )

        raw_sigmoid_score = float(
            torch.sigmoid(
                torch.tensor(raw_logit)
            ).item()
        )

        return {
            "raw_logit": raw_logit,
            "raw_sigmoid_score": raw_sigmoid_score,
            "input_coverage": coverage,
            "imputed_features": imputed_features,
            "model_input": tensor.tolist()[0],}

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> Prediction:
        """Generate a calibrated standalone CPRD prediction."""


        technical_result = self.predict_raw(
            features
        )

        raw_logit = float(
            technical_result["raw_logit"]
        )

        raw_score = float(
            technical_result[
                "raw_sigmoid_score"
            ]
        )

        imputed_features = list(
            technical_result[
                "imputed_features"
            ]
        )

        coverage = float(
            technical_result[
                "input_coverage"
            ]
        )

        calibrated_risk = float(
            self.calibrator.predict_proba(
                np.asarray(
                    [[raw_logit]],
                    dtype=np.float64,
                )
            )[0, 1]
        )

        if not np.isfinite(calibrated_risk):
            raise ValueError(
                "The standalone CPRD calibrator returned "
                "a non-finite value."
            )

        if not 0.0 <= calibrated_risk <= 1.0:
            raise ValueError(
                "The standalone CPRD calibrated probability "
                "is outside the interval [0, 1]."
            )



        if calibrated_risk >= self.threshold:
            category = (
                "At or above the selected research "
                "operating threshold"
            )
        else:
            category = (
                "Below the selected research "
                "operating threshold"
            )

        warnings: List[str] = []

        if imputed_features:
            readable_names = [
                FEATURE_LABELS.get(
                    feature_name,
                    feature_name,
                )
                for feature_name in imputed_features
            ]

            warnings.append(
                "Missing model inputs were handled using "
                "the preprocessing rule applied during "
                "training: "
                + ", ".join(readable_names)
                + "."
            )

        

        return Prediction(
            model_id=self.spec.model_id,
            model_name=self.spec.display_name,
            probability=calibrated_risk,
            threshold=self.threshold,
            category=category,
            input_coverage=coverage,
            imputed_features=imputed_features,
            warnings=warnings,
            contributions=[],
            raw_score=raw_score,
            calibration_method=(
                self.calibration_metadata[
                    "calibration_method"
                ]
            ),
            calibration_population=(
                self.calibration_metadata[
                    "calibration_population"
                ]
            ),
            prediction_horizon_months=int(
                self.calibration_metadata[
                    "prediction_horizon_months"
                ]
            ),
            calibrated=True,
            placeholder=False,
            original_calibrated_probability=(
                original_calibrated_risk
            ),
            experimental_adjustment_factor=(
                EXPERIMENTAL_CPRD_FACTOR
            ),
            experimentally_adjusted=True,
        )
class HospitalSilverPhenotypeMLP(nn.Module):
    """
    Dynamic MLP used by the selected MIMIC-IV and eICU models.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Tuple[int, ...],
        dropout: float,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        previous_dimension = input_dimension

        for hidden_dimension in hidden_dimensions:
            layers.extend(
                [
                    nn.Linear(
                        previous_dimension,
                        hidden_dimension,
                    ),
                    nn.LayerNorm(
                        hidden_dimension
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        dropout
                    ),
                ]
            )

            previous_dimension = (
                hidden_dimension
            )

        layers.append(
            nn.Linear(
                previous_dimension,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class HospitalSilverPhenotypeAdapter:
    """
    Shared research-only inference adapter for the
    MIMIC-IV and eICU silver-phenotype models.
    """

    def __init__(
        self,
        spec: ModelSpec,
        artifact_directory: Path,
    ):
        self.spec = spec

        self.artifact_directory = Path(
            artifact_directory
        )

        self.checkpoint_path = (
            self.artifact_directory
            / "model.pt"
        )

        self.preprocessor_path = (
            self.artifact_directory
            / "preprocessor.joblib"
        )

        self.calibrator_path = (
            self.artifact_directory
            / "calibrator.joblib"
        )

        self.metadata_path = (
            self.artifact_directory
            / "deployment_metadata.json"
        )

        required_paths = [
            self.checkpoint_path,
            self.preprocessor_path,
            self.calibrator_path,
            self.metadata_path,
        ]

        missing_paths = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Missing hospital-model deployment "
                "artifacts: "
                + ", ".join(missing_paths)
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        required_checkpoint_fields = [
            "state_dict",
            "input_dimension",
            "features",
            "hidden_dimensions",
            "dropout",
            "threshold",
            "experiment",
        ]

        missing_checkpoint_fields = [
            field_name
            for field_name
            in required_checkpoint_fields
            if field_name not in checkpoint
        ]

        if missing_checkpoint_fields:
            raise KeyError(
                "Hospital-model checkpoint is "
                "missing: "
                + ", ".join(
                    missing_checkpoint_fields
                )
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            self.metadata = json.load(
                metadata_file
            )

        if (
            self.metadata.get("model_id")
            != self.spec.model_id
        ):
            raise ValueError(
                "Hospital-model metadata ID does "
                "not match the registry ID."
            )

        self.features = list(
            checkpoint["features"]
        )

        metadata_features = list(
            self.metadata.get(
                "features",
                [],
            )
        )

        if metadata_features != self.features:
            raise ValueError(
                "Hospital-model checkpoint and "
                "metadata feature orders differ."
            )

        self.input_dimension = int(
            checkpoint["input_dimension"]
        )

        self.hidden_dimensions = tuple(
            int(value)
            for value
            in checkpoint[
                "hidden_dimensions"
            ]
        )

        self.dropout = float(
            checkpoint["dropout"]
        )

        self.threshold = float(
            checkpoint["threshold"]
        )

        self.constant_zero_features = set(
            self.metadata.get(
                "constant_zero_features",
                [],
            )
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Hospital-model threshold must "
                "lie in [0, 1]."
            )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        self.calibrator = joblib.load(
            self.calibrator_path
        )

        if not hasattr(
            self.preprocessor,
            "transform",
        ):
            raise TypeError(
                "Hospital-model preprocessor "
                "has no transform() method."
            )

        if not hasattr(
            self.calibrator,
            "predict_proba",
        ):
            raise TypeError(
                "Hospital-model calibrator has "
                "no predict_proba() method."
            )

        self.model = (
            HospitalSilverPhenotypeMLP(
                input_dimension=(
                    self.input_dimension
                ),
                hidden_dimensions=(
                    self.hidden_dimensions
                ),
                dropout=self.dropout,
            )
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.to("cpu")
        self.model.eval()

    @staticmethod
    def _age_category(
        value: Any,
    ) -> Tuple[float, bool, bool]:
        """
        Convert raw age into the category used
        during MIMIC/eICU preprocessing.

        Returns:
            category,
            whether age was missing,
            whether age was outside development range.
        """

        try:
            age = float(value)

            if not np.isfinite(age):
                raise ValueError

        except (TypeError, ValueError):
            return 0.0, True, False

        outside_development_range = (
            age < 50.0
            or age > 80.0
        )

        age = float(
            np.clip(
                age,
                50.0,
                80.0,
            )
        )

        if age < 55.0:
            category = 0.0
        elif age < 60.0:
            category = 1.0
        elif age < 65.0:
            category = 2.0
        elif age < 70.0:
            category = 3.0
        elif age < 75.0:
            category = 4.0
        else:
            category = 5.0

        return (
            category,
            False,
            outside_development_range,
        )

    @staticmethod
    def _bmi_category(
        value: Any,
        available: Any,
    ) -> Tuple[float, bool]:
        """
        Convert raw BMI into the category used
        during MIMIC/eICU preprocessing.
        """

        if available is False:
            return 0.0, True

        try:
            bmi = float(value)

            if (
                not np.isfinite(bmi)
                or bmi <= 0.0
            ):
                raise ValueError

        except (TypeError, ValueError):
            return 0.0, True

        if bmi < 18.5:
            category = 1.0
        elif bmi < 25.0:
            category = 2.0
        elif bmi < 30.0:
            category = 3.0
        else:
            category = 4.0

        return category, False

    @staticmethod
    def _binary_value(
        value: Any,
    ) -> Tuple[float, bool]:
        """
        Convert a UI value into zero or one.

        Returns:
            converted value,
            whether missing-value handling was used.
        """

        if value is None:
            return 0.0, True

        if isinstance(value, str):
            normalised = (
                value.strip().lower()
            )

            if normalised in {
                "yes",
                "true",
                "1",
                "1.0",
            }:
                return 1.0, False

            if normalised in {
                "no",
                "false",
                "0",
                "0.0",
            }:
                return 0.0, False

            return 0.0, True

        try:
            numeric_value = float(value)

            if not np.isfinite(
                numeric_value
            ):
                return 0.0, True

            if numeric_value > 0.0:
                return 1.0, False

            return 0.0, False

        except (TypeError, ValueError):
            return 0.0, True

    def _prepare_input(
        self,
        patient_record: Mapping[str, Any],
    ) -> Tuple[
        torch.Tensor,
        List[str],
        float,
        List[str],
    ]:
        """
        Construct the exact preprocessed tensor expected
        by the selected MIMIC/eICU model.
        """

        imputed_features: List[str] = []
        preparation_warnings: List[str] = []

        (
            age_category,
            age_imputed,
            age_outside_range,
        ) = self._age_category(
            patient_record.get("age")
        )

        if age_imputed:
            imputed_features.append(
                "age"
            )

        if age_outside_range:
            preparation_warnings.append(
                "Age was outside the 50–80-year "
                "development range and was clipped "
                "for this research-model calculation."
            )

        (
            bmi_category,
            bmi_imputed,
        ) = self._bmi_category(
            patient_record.get(
                "bmifinal2"
            ),
            patient_record.get(
                "bmi_available",
                True,
            ),
        )

        if bmi_imputed:
            imputed_features.append(
                "bmifinal2"
            )

        recorded_sex = str(
            patient_record.get(
                "sex_recorded",
                "",
            )
        ).strip().lower()

        if recorded_sex == "male":
            gender_value = 0.0

        elif recorded_sex == "female":
            gender_value = 1.0

        else:
            # The training imputer's most frequent
            # value was 1.
            gender_value = 1.0

            imputed_features.append(
                "sex_recorded"
            )

        row: Dict[str, float] = {}

        for feature_name in self.features:
            if (
                feature_name
                in self.constant_zero_features
            ):
                # This variable did not vary in the
                # selected model's training subset.
                row[feature_name] = 0.0

            elif feature_name == "age":
                row[feature_name] = (
                    age_category
                )

            elif feature_name == "bmifinal2":
                row[feature_name] = (
                    bmi_category
                )

            elif feature_name == "gender_1":
                row[feature_name] = (
                    gender_value
                )

            else:
                (
                    value,
                    was_imputed,
                ) = self._binary_value(
                    patient_record.get(
                        feature_name
                    )
                )

                row[feature_name] = value

                if was_imputed:
                    imputed_features.append(
                        feature_name
                    )

        raw_frame = pd.DataFrame(
            [row],
            columns=self.features,
        )

        transformed = np.asarray(
            self.preprocessor.transform(
                raw_frame
            ),
            dtype=np.float32,
        )

        expected_shape = (
            1,
            self.input_dimension,
        )

        if transformed.shape != expected_shape:
            raise ValueError(
                "Hospital-model preprocessor "
                "returned shape "
                f"{transformed.shape}; expected "
                f"{expected_shape}."
            )

        if not np.isfinite(
            transformed
        ).all():
            raise ValueError(
                "Hospital-model input contains "
                "non-finite values."
            )

        usable_features = [
            feature_name
            for feature_name in self.features
            if (
                feature_name
                not in self.constant_zero_features
            )
        ]

        missing_usable_features = {
            feature_name
            for feature_name
            in imputed_features
            if (
                feature_name
                in usable_features
                or feature_name
                in {
                    "sex_recorded",
                    "bmifinal2",
                    "age",
                }
            )
        }

        observed_count = (
            len(usable_features)
            - len(
                missing_usable_features
            )
        )

        coverage = max(
            0.0,
            min(
                1.0,
                observed_count
                / max(
                    len(usable_features),
                    1,
                ),
            ),
        )

        tensor = torch.from_numpy(
            transformed
        )

        return (
            tensor,
            sorted(
                set(imputed_features)
            ),
            coverage,
            preparation_warnings,
        )

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> Prediction:
        """
        Generate an internally calibrated research-only
        silver-phenotype estimate.
        """

        (
            tensor,
            imputed_features,
            coverage,
            warnings,
        ) = self._prepare_input(
            features
        )

        with torch.inference_mode():
            raw_logit = float(
                self.model(tensor)[0].item()
            )

        raw_score = float(
            torch.sigmoid(
                torch.tensor(
                    raw_logit
                )
            ).item()
        )

        # Reproduce the exact calibration transformation
        # used by the training pipeline.
        clipped_score = float(
            np.clip(
                raw_score,
                1e-6,
                1.0 - 1e-6,
            )
        )

        calibration_logit = math.log(
            clipped_score
            / (
                1.0
                - clipped_score
            )
        )

        calibrated_probability = float(
            self.calibrator.predict_proba(
                np.asarray(
                    [[calibration_logit]],
                    dtype=np.float64,
                )
            )[0, 1]
        )

        if not np.isfinite(
            calibrated_probability
        ):
            raise ValueError(
                "Hospital-model calibrator returned "
                "a non-finite value."
            )

        if not (
            0.0
            <= calibrated_probability
            <= 1.0
        ):
            raise ValueError(
                "Hospital-model calibrated value "
                "is outside [0, 1]."
            )

        if (
            calibrated_probability
            >= self.threshold
        ):
            category = (
                "At or above the internal "
                "research operating threshold"
            )
        else:
            category = (
                "Below the internal research "
                "operating threshold"
            )

        warnings.extend(
            [
                (
                    "This model estimates a derived "
                    "silver-label phenotype in a "
                    "hospital or critical-care population."
                ),
                (
                    "It is shown for research comparison "
                    "only and is not a confirmed "
                    "lung-cancer probability or a "
                    "clinically validated referral tool."
                ),
            ]
        )

        return Prediction(
            model_id=self.spec.model_id,
            model_name=self.spec.display_name,
            probability=(
                calibrated_probability
            ),
            threshold=self.threshold,
            category=category,
            input_coverage=coverage,
            imputed_features=(
                imputed_features
            ),
            warnings=warnings,
            contributions=[],
            raw_score=raw_score,
            calibration_method=str(
                self.metadata.get(
                    "calibration_method",
                    (
                        "Platt logistic "
                        "calibration"
                    ),
                )
            ),
            calibration_population=str(
                self.metadata.get(
                    "development_population",
                    self.spec.population,
                )
            ),
            prediction_horizon_months=None,
            calibrated=True,
            placeholder=False,
        )
class HarmonisedMLP(nn.Module):
    """Exact MLP used by the CPRD–VARHA experiment."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x).squeeze(1)

class CPRDVARHAModelAdapter:
    """
    Inference adapter for the adaptive CPRD–VARHA
    federated model.
    """

    REQUIRED_FEATURES = [
        "age",
        "copd_emphysema",
        "ckd",
        "cerebrovascular",
        "cardiovascular",
        "liver",
    ]

    def __init__(
    self,
    spec: ModelSpec,
    checkpoint_path: Path,
    calibrator_path: Path,
    calibration_metadata_path: Path,):
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.calibrator_path = Path(calibrator_path)
        self.calibration_metadata_path = Path(
            calibration_metadata_path
    )

        # -----------------------------------------------
        # Confirm that the checkpoint exists
        # -----------------------------------------------
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "CPRD–VARHA checkpoint was not found at: "
                f"{self.checkpoint_path}"
            )

        # -----------------------------------------------
        # Load the checkpoint on CPU
        # -----------------------------------------------
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "The CPRD–VARHA checkpoint should contain "
                "a dictionary, but a different object was found."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "The CPRD–VARHA checkpoint does not contain "
                "'model_state_dict'."
            )

        # -----------------------------------------------
        # Verify feature names and order
        # -----------------------------------------------
        checkpoint_features = checkpoint.get("features")

        if checkpoint_features != self.REQUIRED_FEATURES:
            raise ValueError(
                "The checkpoint feature schema does not match "
                "the expected CPRD–VARHA schema. "
                f"Checkpoint features: {checkpoint_features}. "
                f"Expected features: {self.REQUIRED_FEATURES}."
            )

        # -----------------------------------------------
        # Retrieve preprocessing information
        # -----------------------------------------------
        self.age_center = float(
            checkpoint.get("age_center", 65.0)
        )

        self.age_scale = float(
            checkpoint.get("age_scale", 10.0)
        )

                # -----------------------------------------------
        # Load the fitted probability calibrator
        # -----------------------------------------------
        if not self.calibrator_path.exists():
            raise FileNotFoundError(
                "The CPRD–VARHA probability calibrator was "
                "not found at: "
                f"{self.calibrator_path}"
            )
        
        self.calibrator = joblib.load(
            self.calibrator_path
        )
        
        if not hasattr(self.calibrator, "predict_proba"):
            raise TypeError(
                "The loaded calibrator does not provide "
                "a predict_proba() method."
            )
        
        # -----------------------------------------------
        # Load calibration metadata
        # -----------------------------------------------
        if not self.calibration_metadata_path.exists():
            raise FileNotFoundError(
                "The calibration metadata file was not "
                "found at: "
                f"{self.calibration_metadata_path}"
            )
        
        with self.calibration_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            self.calibration_metadata = json.load(
                metadata_file
            )
        
        required_calibration_fields = [
            "calibration_method",
            "calibration_input",
            "calibration_population",
            "outcome_definition",
            "prediction_horizon_months",
            "calibrated_threshold",
        ]
        
        missing_calibration_fields = [
            field_name
            for field_name in required_calibration_fields
            if field_name not in self.calibration_metadata
        ]
        
        if missing_calibration_fields:
            raise KeyError(
                "Calibration metadata is missing: "
                + ", ".join(missing_calibration_fields)
            )
        
        if (
            self.calibration_metadata["calibration_input"]
            != "raw_logit"
        ):
            raise ValueError(
                "This application expects a calibrator fitted "
                "using raw model logits."
            )
        
        self.threshold = float(
            self.calibration_metadata[
                "calibrated_threshold"
            ]
        )
        
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "The calibrated threshold must be between "
                "0 and 1."
            )
        


        # -----------------------------------------------
        # Reconstruct and load the neural network
        # -----------------------------------------------
        self.model = HarmonisedMLP()

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to("cpu")
        self.model.eval()

        # Retain deployment metadata
        self.metadata = {
            "features": checkpoint_features,
            "target": checkpoint.get("target"),
            "architecture": checkpoint.get(
                "architecture"
            ),
            "aggregation": checkpoint.get(
                "aggregation"
            ),
            "best_global_round": checkpoint.get(
                "best_global_round"
            ),
            "best_validation_mean_AUROC": checkpoint.get(
                "best_validation_mean_AUROC"
            ),
            "raw_CPRD_validation_threshold": checkpoint.get(
            "CPRD_validation_threshold"
        ),
        "calibrated_threshold": self.threshold,
        "calibration_method": self.calibration_metadata[
            "calibration_method"
        ],
        "calibration_population": self.calibration_metadata[
            "calibration_population"
        ],
        "outcome_definition": self.calibration_metadata[
            "outcome_definition"
        ],
        "prediction_horizon_months": self.calibration_metadata[
            "prediction_horizon_months"
        ],
            "VARHA_validation_threshold": checkpoint.get(
                "VARHA_validation_threshold"
            ),
            "age_center": self.age_center,
            "age_scale": self.age_scale,
        }

    @staticmethod
    def _binary_value(
        value: Any,
    ) -> Tuple[float, bool]:
        """
        Convert a clinical value to the binary representation
        used during training.

        Returns:
            transformed value,
            whether missing-value handling was applied.
        """

        if value is None:
            return 0.0, True

        if isinstance(value, str):
            normalised = value.strip().lower()

            if normalised in {
                "yes",
                "true",
                "1",
                "1.0",
            }:
                return 1.0, False

            if normalised in {
                "no",
                "false",
                "0",
                "0.0",
            }:
                return 0.0, False

            return 0.0, True

        try:
            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                return 0.0, True

            if numeric_value > 0:
                return 1.0, False

            return 0.0, False

        except (TypeError, ValueError):
            return 0.0, True

    def _prepare_input(
        self,
        patient_record: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, List[str], float]:
        """
        Apply the exact preprocessing used in the training
        script and construct a 1 x 6 tensor.
        """

        imputed_features: List[str] = []

        # -----------------------------------------------
        # Age transformation
        # -----------------------------------------------
        raw_age = patient_record.get("age")

        try:
            age = float(raw_age)

            if not np.isfinite(age):
                raise ValueError

        except (TypeError, ValueError):
            age = self.age_center
            imputed_features.append("age")

        age = float(
            np.clip(
                age,
                18.0,
                110.0,
            )
        )

        transformed_age = (
            age - self.age_center
        ) / self.age_scale

        transformed_values = [
            transformed_age
        ]

        # -----------------------------------------------
        # Binary comorbidities
        # -----------------------------------------------
        for feature_name in self.REQUIRED_FEATURES[1:]:
            value, was_imputed = self._binary_value(
                patient_record.get(feature_name)
            )

            transformed_values.append(value)

            if was_imputed:
                imputed_features.append(
                    feature_name
                )

        # -----------------------------------------------
        # Calculate model-input coverage
        # -----------------------------------------------
        observed_count = (
            len(self.REQUIRED_FEATURES)
            - len(imputed_features)
        )

        coverage = (
            observed_count
            / len(self.REQUIRED_FEATURES)
        )

        tensor = torch.tensor(
            [transformed_values],
            dtype=torch.float32,
        )

        return (
            tensor,
            imputed_features,
            coverage,
        )

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> Prediction:
        """
        Generate the CPRD–VARHA model score for one patient.
        """

        (
            tensor,
            imputed_features,
            coverage,
        ) = self._prepare_input(features)

        # -----------------------------------------------
        # Frozen-model inference
        # -----------------------------------------------
        with torch.inference_mode():
            raw_logit = float(
                self.model(tensor)[0].item()
            )
        
        # Keep the raw sigmoid score internally.
        # It is not shown as a cancer probability.
        raw_score = float(
            torch.sigmoid(
                torch.tensor(raw_logit)
            ).item()
        )
    
        # -----------------------------------------------
        # Convert the raw logit to calibrated cancer risk
        # -----------------------------------------------
        calibrated_risk = float(
            self.calibrator.predict_proba(
                np.asarray(
                    [[raw_logit]],
                    dtype=np.float64,
                )
            )[0, 1]
        )
        
        if not np.isfinite(calibrated_risk):
            raise ValueError(
                "The calibrator returned a non-finite value."
            )
        
        if not 0.0 <= calibrated_risk <= 1.0:
            raise ValueError(
                "The calibrated probability is outside "
                "the interval [0, 1]."
            )
        
        # -----------------------------------------------
        # Apply the calibrated operating threshold
        # -----------------------------------------------
        if calibrated_risk >= self.threshold:
            category = (
                "At or above the selected investigation "
                "threshold"
            )
        else:
            category = (
                "Below the selected investigation threshold"
            )

        warnings: List[str] = []

        if imputed_features:
            readable_names = [
                FEATURE_LABELS.get(
                    feature_name,
                    feature_name,
                )
                for feature_name in imputed_features
            ]

            warnings.append(
                "Missing model inputs were handled using "
                "the preprocessing rule applied during "
                "training: "
                + ", ".join(readable_names)
                + "."
            )


        return Prediction(
            model_id=self.spec.model_id,
            model_name=self.spec.display_name,
            probability=calibrated_risk,
            threshold=self.threshold,
            category=category,
            input_coverage=float(coverage),
            imputed_features=imputed_features,
            warnings=warnings,
            contributions=[],
            raw_score=raw_score,
            calibration_method=self.calibration_metadata[
                "calibration_method"
            ],
            calibration_population=self.calibration_metadata[
                "calibration_population"
            ],
            prediction_horizon_months=int(
                self.calibration_metadata[
                    "prediction_horizon_months"
                ]
            ),
            calibrated=True,
            placeholder=False,
        )





class PlaceholderRiskModel:
    """Deterministic mock inference adapter. Never use for clinical decisions."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @staticmethod
    def _present(value: Any) -> float:
        return 1.0 if value in (True, "Yes", "Current", "Former") else 0.0

    def predict(self, features: Mapping[str, Any]) -> Prediction:
        age = float(features.get("age", 50))
        bmi = float(features.get("bmifinal2", 25.0))
        contributions = {
            "Age": max(0.0, age - 40.0) * 0.030,
            "Smoking exposure": self._present(features.get("smk_qt_final_2_2")) * 0.90,
            "Haemoptysis": self._present(features.get("haemoptysisoneyear")) * 1.15,
            "Unexplained weight loss": self._present(features.get("weightlossoneyear")) * 0.80,
            "Breathlessness": self._present(features.get("dyspnoeaoneyear")) * 0.38,
            "COPD": self._present(features.get("copd_com")) * 0.52,
            "Family history of cancer": self._present(features.get("familyhcancer_1")) * 0.30,
            "Persistent cough": self._present(features.get("persistent_cough")) * 0.42,
            "BMI outside reference range": 0.20 if bmi < 18.5 else 0.0,
        }
        # Stable, tiny separation makes all four placeholder cards visibly distinct.
        digest = hashlib.sha256(self.spec.model_id.encode()).digest()[0]
        model_offset = (digest / 255.0 - 0.5) * 0.20
        logit = -4.75 + sum(contributions.values()) + model_offset
        probability = 1.0 / (1.0 + math.exp(-logit))
        probability = min(max(probability, 0.001), 0.75)

        if probability >= self.spec.threshold:
            category = "Elevated predicted risk"
        elif probability >= self.spec.threshold * 0.60:
            category = "Intermediate predicted risk"
        else:
            category = "Lower predicted risk"

        warnings: List[str] = ["Demonstration score only — placeholder model in use."]
        if self.spec.model_id in {"mimic", "eicu"}:
            warnings.append("Development population differs from the intended GP setting.")

        ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        ranked = [(name, value) for name, value in ranked if value > 0][:5]
        return Prediction(
            model_id=self.spec.model_id,
            model_name=self.spec.display_name,
            probability=probability,
            threshold=self.spec.threshold,
            category=category,
            input_coverage=1.0,
            imputed_features=[],
            warnings=warnings,
            contributions=ranked,
        )


@st.cache_resource

def load_model_registry() -> Dict[str, Any]:
    app_directory = Path(__file__).resolve().parent
        
    return {
        "cprd": CPRDStandaloneModelAdapter(
    spec=MODEL_SPECS["cprd"],
    checkpoint_path=(
        app_directory
        / "models"
        / "cprd"
        / "standalone_CPRD_same_MLP.pt"
    ),
    calibrator_path=(
        app_directory
        / "models"
        / "cprd"
        / "calibrator.joblib"
    ),
    calibration_metadata_path=(
        app_directory
        / "models"
        / "cprd"
        / "calibration_metadata.json"
    ),),

        "cprd_varha": CPRDVARHAModelAdapter(
            spec=MODEL_SPECS["cprd_varha"],
            checkpoint_path=(
                app_directory
                / "models"
                / "cprd_varha"
                / "final_adaptive_CPRD_VARHA_global_model.pt"
            ),
            calibrator_path=(
                app_directory
                / "models"
                / "cprd_varha"
                / "calibrator.joblib"
            ),
            calibration_metadata_path=(
                app_directory
                / "models"
                / "cprd_varha"
                / "calibration_metadata.json"
            ),
        ),

                "mimic": HospitalSilverPhenotypeAdapter(
            spec=MODEL_SPECS["mimic"],
            artifact_directory=(
                app_directory
                / "models"
                / "mimic"
            ),
        ),

        "eicu": HospitalSilverPhenotypeAdapter(
            spec=MODEL_SPECS["eicu"],
            artifact_directory=(
                app_directory
                / "models"
                / "eicu"
            ),
        ),
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem;}
        /* Keep the navigation readable when the viewer uses Streamlit's dark
           theme. The earlier pale sidebar inherited white theme text. */
        section[data-testid="stSidebar"] {background: #132129 !important;}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #F4F8F8 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #0D141B !important;
            border-color: #547078 !important;
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #FFF4C7 !important;
            border-color: #E1B93E !important;
        }
        section[data-testid="stSidebar"] [data-testid="stAlert"] p,
        section[data-testid="stSidebar"] [data-testid="stAlert"] div {
            color: #352A00 !important;
        }
        .hero {
    background: linear-gradient(125deg,#092F3A,#14636B);
    color: white;
    border-radius: 16px;
    padding: 16px 26px;
    margin-top: 8px;
    margin-bottom: 16px;
}
        .hero h1 {
    font-size: 1.75rem;
    line-height: 1.2;
    margin: 0 0 8px 0;
    color: white;
}
.hero p {
    font-size: 1rem;
    line-height: 1.4;
    margin: 0;
    opacity: 0.88;
}
        .hero p {margin:0; opacity:.88;}
        .clinical-card {border:1px solid #DCE7E7; border-radius:14px; padding:18px;
                        background:white; box-shadow:0 2px 10px rgba(19,62,67,.05);}
        .risk-number {
    font-size: 2.4rem;
    font-weight: 750;
    line-height: 1.15;
    color: #7ED6D3;
    margin: 6px 0 10px 0;
}
        .eyebrow {font-size:.76rem; font-weight:700; letter-spacing:.08em;
                  text-transform:uppercase; color:#47727A;}
        .safety {background:#FFF4E5; border-left:5px solid #D97800; padding:14px 16px;
                 border-radius:8px; margin:10px 0;}
        .prototype {
    background: #DCEBFA;
    color: #172B3A !important;
    border-left: 5px solid #3156A3;
    padding: 12px 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.prototype strong {
    color: #172B3A !important;
}
        .muted {color:#61747A; font-size:.9rem;}
        .project-strip {border:1px solid #284650; background:#101A21; border-radius:14px;
                        padding:13px 18px; margin-bottom:14px;}
        .project-kicker {font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
                         color:#86D7D4; font-weight:750;}
        .project-name {font-size:1rem; color:#F3F8F8; margin-top:3px;}
        .footer-card {border-top:1px solid #38515A; margin-top:30px; padding-top:18px;
                      color:#9FB0B4; font-size:.82rem;}
        div[data-testid="stMetric"] {border:1px solid #DCE7E7; padding:12px;
                                     border-radius:12px; background:white;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    defaults = {
        "assessment_complete": False,
        "predictions": {},
        "patient_features": {},
        "clinical_notes": {},
        "decision": {},
        "case_id": f"DEMO-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def first_existing_asset(*names: str) -> Optional[Path]:
    """Return the first locally available project asset."""
    for name in names:
        path = Path(__file__).resolve().parent / name
        if path.exists():
            return path
    return None


def render_project_header() -> None:
    phase_logo = first_existing_asset("PhaseIVAIHorizontal.png", "phase_iv_ai_logo.png")
    ntu_logo = first_existing_asset("NTUHorizontal.png", "ntu_logo.png")
    eu_flag = first_existing_asset("EUflag.png", "eu_flag.png")

    st.markdown(
    "<div style='height: 30px;'></div>",
    unsafe_allow_html=True,
)

    # logo_left, logo_middle, logo_right = st.columns([2.2, 1.1, 0.7])
    logo_left, logo_middle, logo_right = st.columns(3)
    with logo_left:
        if phase_logo:
            st.image(str(phase_logo), width=240)
        else:
            st.markdown("### PHASE IV AI")
    with logo_middle:
        if ntu_logo:
            st.image(str(ntu_logo), width=240)
    with logo_right:
        if eu_flag:
            st.image(str(eu_flag), width=240)
   

  
    st.markdown(
        f'<div class="hero"><h1>{APP_TITLE}</h1>'
        '<p>Primary-care decision support for assessing lung-cancer risk in patients presenting with respiratory symptoms</p></div>',
        unsafe_allow_html=True,
    )


def render_project_footer() -> None:
    st.markdown(
        f'<div class="footer-card"><strong>PHASE IV AI · Project {PROJECT_NUMBER}</strong><br>'
        'Funded by the European Union. Views and opinions expressed are those of the author(s) only and do not '
        'necessarily reflect those of the European Union or the European Health and Digital Executive Agency '
        '(HaDEA). Neither the European Union nor the granting authority can be held responsible for them.<br><br>'
        f'<a href="{PROJECT_URL}" target="_blank">Visit the PHASE IV AI project website</a></div>',
        unsafe_allow_html=True,
    )


def yes_no(label: str, key: str, help_text: Optional[str] = None) -> bool:
    return st.radio(label, ["No", "Yes"], horizontal=True, key=key, help=help_text) == "Yes"


def risk_badge(category: str) -> str:
    if category.startswith("Elevated"):
        return "🔴"
    if category.startswith("Intermediate"):
        return "🟠"
    if category.startswith("Lower"):
        return "🟢"
    return "⚪"


def collect_assessment() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    st.subheader("Patient assessment")
    st.caption("Enter information available during the current GP consultation. Do not enter direct identifiers in this prototype.")

    with st.expander("1 · Patient profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 100, 58, 1)
        sex = c2.selectbox("Sex recorded in the health record", ["Female", "Male", "Other / not recorded"])
        bmi_known = c3.checkbox("BMI available", value=True)
        bmi = c3.number_input("BMI (kg/m²)", 10.0, 70.0, 24.5, 0.1, disabled=not bmi_known)

    with st.expander("2 · Current presentation", expanded=True):
        c1, c2 = st.columns(2)
        complaint = c1.selectbox(
            "Main reason for consultation",
            ["Persistent cough", "Breathlessness", "Chest discomfort", "Haemoptysis", "Weight loss", "Other respiratory concern"],
        )
        duration = c2.selectbox("Approximate symptom duration", ["Less than 2 weeks", "2–3 weeks", "4–6 weeks", "More than 6 weeks", "Unknown"])
        c1, c2 = st.columns(2)
        persistent_cough = yes_no("Persistent or changing cough", "persistent_cough")
        dyspnoea = yes_no("Breathlessness during the previous year", "dyspnoea")
        haemoptysis = yes_no("Haemoptysis (coughing up blood) during the previous year", "haemoptysis")
        sputum = yes_no("Change in sputum", "sputum")
        weight_loss = yes_no("Unexplained weight loss during the previous year", "weight_loss")
        recurrent_infection = yes_no(
    "Unresolved or recurrent chest infection during the previous 12 months",
    "recurrent_infection",
    help_text=(
        "Select Yes if the patient has had a clinician-diagnosed chest "
        "infection that did not resolve as expected following treatment, "
        "or repeated clinician-diagnosed chest infections during the "
        "previous 12 months."
    ),
)

    with st.expander("3 · Lifestyle and respiratory history"):
        c1, c2 = st.columns(2)
    
        smoking = c1.selectbox(
            "Smoking status",
            ["Never", "Former", "Current", "Unknown"],
        )
    
        smoking_duration_years = c1.number_input(
            "Duration of smoking (years)",
            min_value=0,
            max_value=90,
            value=0,
            step=1,
            help=(
                "Total number of years during which the patient smoked. "
                "Enter 0 for a never-smoker."
            ),
        )
    
        average_cigarettes_day = c1.number_input(
            "Average cigarettes smoked per day",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help=(
                "Average number of cigarettes smoked daily during the "
                "patient's smoking period."
            ),
        )
    
        years_since_quitting = c1.number_input(
            "Years since quitting",
            min_value=0,
            max_value=90,
            value=0,
            step=1,
            help=(
                "Complete this field for former smokers. Enter 0 for "
                "current smokers and never-smokers."
            ),
        )
    
        calculated_pack_years = (
            average_cigarettes_day / 20.0
        ) * smoking_duration_years
    
        c1.metric(
            "Calculated pack-years",
            f"{calculated_pack_years:.1f}",
            help=(
                "Pack-years = average cigarette packs smoked per day "
                "multiplied by years of smoking. One pack is treated as "
                "20 cigarettes."
            ),
        )
    
        alcohol_units_week = c2.number_input(
            "Alcohol consumption (units per week)",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0,
            help=(
                "One UK alcohol unit equals 10 ml or 8 g of pure alcohol."
            ),
        )
    
        if alcohol_units_week == 0:
            alcohol_category = "None"
        elif alcohol_units_week <= 14:
            alcohol_category = "1–14 units/week"
        else:
            alcohol_category = "15 or more units/week"
    
        c2.caption(
            f"Alcohol category: **{alcohol_category}**"
        )
    
        copd = yes_no(
            "COPD diagnosis",
            "copd",
        )
            

    with st.expander("4 · Personal and family medical history"):
        st.markdown("##### Conditions used by the CPRD models")
    
        c1, c2 = st.columns(2)

    with c1:
        ckd = yes_no(
            "Chronic kidney disease",
            "ckd",
        )

        cerebrovascular = yes_no(
            "Cerebrovascular disease",
            "cerebrovascular",
            help_text=(
                "Select Yes when cerebrovascular disease is "
                "recorded in the patient's medical history."
            ),
        )

    with c2:
        cardiovascular = yes_no(
            (
                "Ischaemic heart disease or previous "
                "myocardial infarction"
            ),
            "cardiovascular",
            help_text=(
                "This corresponds to the harmonised "
                "cardiovascular feature used by the "
                "CPRD–VARHA model."
            ),
        )

        liver_disease = yes_no(
            "Liver disease",
            "liver_disease",
        )

    st.divider()
    st.markdown("##### Other medical and family history")

    c1, c2 = st.columns(2)

    with c1:
        family_lung = yes_no(
            "Family history of lung cancer",
            "family_lung",
        )

        family_cancer = yes_no(
            "Family history of another cancer",
            "family_cancer",
        )

        previous_cancer = st.multiselect(
            "Previous malignancies",
            [
                "Thyroid",
                "Stomach",
                "Kidney",
                "Ovarian",
                "Leukaemia",
                "Myeloma",
                "Melanoma",
                "Head/neck",
                "Bladder",
                "Pancreatic",
            ],
        )

    with c2:
        other_history = st.text_area(
            "Other relevant history",
            placeholder="Optional clinical context",
        )

    with st.expander("5 · Functional status and additional information"):
        performance = st.selectbox("ECOG performance status",[
                "Not recorded","0 — Fully active; no restriction","1 — Restricted in strenuous activity but ambulatory","2 — Ambulatory and capable of self-care; unable to work",
                "3 — Limited self-care; in bed or chair for more than 50% of the day","4 — Completely disabled; totally confined to bed or chair",],
            help=("Select the category that best represents the patient's " "functional status at the time of assessment."), )
    
        clinical_notes = st.text_area(
            "Additional clinical notes",
            placeholder="Do not include direct patient identifiers",
        )
    
        features: Dict[str, Any] = {
            "age": age,
            "sex_recorded": sex,
            "gender_1": sex == "Male",
            "smk_qt_final_2_2": smoking,
            "alc_units_day_7": alcohol_units_week,
            "bmifinal2": bmi if bmi_known else 25.0,
            "bmi_available": bmi_known,
            "famhlg_1": family_lung or family_cancer,
            "familyhcancer_1": family_cancer,
            "thyroid_ca_1": "Thyroid" in previous_cancer,
            "stomach_ca_1": "Stomach" in previous_cancer,
            "kidney_ca_1": "Kidney" in previous_cancer,
            "copd_com": copd,
            "copd_emphysema": copd,
            "ckd": ckd,
            "cerebrovascular": cerebrovascular,
            "cardiovascular": cardiovascular,
            "liver": liver_disease,
            "ovary_ca_1": "Ovarian" in previous_cancer,
            "lek_1": "Leukaemia" in previous_cancer,
            "myeloma_1": "Myeloma" in previous_cancer,
            "melanoma_1": "Melanoma" in previous_cancer,
            "hdnk_ca_1": "Head/neck" in previous_cancer,
            "bladder_ca_1": "Bladder" in previous_cancer,
            "pancreas_1": "Pancreatic" in previous_cancer,
            "dyspnoeaoneyear": dyspnoea,
            "haemoptysisoneyear": haemoptysis,
            "sputumoneyear": sputum,
            "weightlossoneyear": weight_loss,
            "persistent_cough": persistent_cough,
        }
        context = {
            "main_complaint": complaint,
            "duration": duration,
            "smoking_duration_years": smoking_duration_years,
            "average_cigarettes_per_day": average_cigarettes_day,
            "years_since_quitting": (
                years_since_quitting
                if smoking == "Former"
                else 0
            ),
            "calculated_pack_years": calculated_pack_years,
            "alcohol_units_per_week": alcohol_units_week,
            "alcohol_category": alcohol_category,
            "recurrent_infection": recurrent_infection,
            "other_history": other_history,
            "functional_status": performance,
            "consultation_notes": clinical_notes,
            "bmi_was_imputed": not bmi_known,
        }
    return features, context


def validate(features: Mapping[str, Any], context: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if context["main_complaint"] == "Other respiratory concern" and not context["consultation_notes"].strip():
        warnings.append("Consider recording the presenting concern in the consultation notes.")
    if features["smk_qt_final_2_2"] == "Unknown":
        warnings.append("Smoking history is unknown; model reliability may be reduced.")
    if context["bmi_was_imputed"]:
        warnings.append("BMI is unavailable and will be handled by the model-specific missing-data pipeline.")
    if not any(
        bool(features[name])
        for name in ["persistent_cough", "dyspnoeaoneyear", "haemoptysisoneyear", "sputumoneyear", "weightlossoneyear"]
    ):
        warnings.append("None of the structured respiratory symptoms are marked as present.")
    if float(features["age"]) < 30:
        warnings.append("The patient may be outside the main age distribution of the development data.")
    return errors, warnings


def render_safety_check(features: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    flags = []
    if features.get("haemoptysisoneyear"):
        flags.append("haemoptysis")
    if features.get("weightlossoneyear"):
        flags.append("unexplained weight loss")
    if flags:
        st.markdown(
            '<div class="safety"><strong>Clinical warning features recorded</strong><br>'
            + ", ".join(flags).capitalize()
            + ". A low model score must not delay an appropriate clinical pathway.</div>",
            unsafe_allow_html=True,
        )
        return True
    st.success("No configured warning feature was identified. Continue to apply clinical judgement.")
    return False


# def run_models(features: Mapping[str, Any], selected_ids: List[str]) -> Dict[str, Prediction]:
#     return {model_id: MODEL_REGISTRY[model_id].predict(features) for model_id in selected_ids}

def run_models(
    features: Mapping[str, Any],
    selected_ids: List[str],
) -> Dict[str, Prediction]:
    registry = load_model_registry()

    return {
        model_id: registry[model_id].predict(features)
        for model_id in selected_ids
    }
def render_risk_scale(
    probability: float,
    threshold: float,
) -> None:
    """Display risk relative to the model's research threshold."""

    probability = float(probability)
    threshold = float(threshold)

    if threshold <= 0:
        st.warning("A valid research operating threshold is unavailable.")
        return

    lower_boundary = 0.035
    higher_boundary = 0.065
    scale_maximum = 0.10

    marker_position = min(
        max((probability / scale_maximum) * 100.0, 0.0),
        100.0,
    )

    lower_boundary_position = (
        lower_boundary / scale_maximum
    ) * 100.0

    threshold_position = (
        higher_boundary / scale_maximum
    ) * 100.0

    

    if probability < lower_boundary:
        zone_name = "Lower"
        zone_colour = "#22c55e"
    elif probability < higher_boundary:
        zone_name = "Intermediate"
        zone_colour = "#f59e0b"
    else:
        zone_name = "Higher"
        zone_colour = "#ef4444"

    st.html(
        dedent(
            f"""

        <div style="margin-top:0.75rem; margin-bottom:0.25rem;">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:0.45rem;
                font-size:0.92rem;
            ">
                <span>
                    <strong>Position relative to research threshold</strong>
                </span>

                <span style="
                    color:{zone_colour};
                    font-weight:700;
                ">
                    {zone_name}
                </span>
            </div>

            <div style="
                position:relative;
                width:100%;
                height:18px;
                border-radius:9px;
                background:linear-gradient(
                    to right,
                    #22c55e 0%,
                    #22c55e {lower_boundary_position:.2f}%,
                    #f59e0b {lower_boundary_position:.2f}%,
                    #f59e0b {threshold_position:.2f}%,
                    #ef4444 {threshold_position:.2f}%,
                    #ef4444 100%
                );
                box-shadow:inset 0 0 0 1px
                    rgba(255,255,255,0.20);
            ">

                <div style="
                    position:absolute;
                    left:{marker_position:.2f}%;
                    top:-6px;
                    width:4px;
                    height:30px;
                    background:white;
                    border:1px solid #111827;
                    border-radius:2px;
                    transform:translateX(-2px);
                    box-shadow:0 1px 4px rgba(0,0,0,0.50);
                "></div>

            </div>

            <div style="
                position:relative;
                height:24px;
                margin-top:5px;
                color:#9ca3af;
                font-size:0.75rem;
            ">

                <span style="
                    position:absolute;
                    left:0;
                ">
                    Lower
                </span>

                <span style="
                    position:absolute;
                    left:{lower_boundary_position:.2f}%;
                    transform:translateX(-50%);
                ">
                    Intermediate
                </span>

                <span style="
                    position:absolute;
                    left:{threshold_position:.2f}%;
                    transform:translateX(-50%);
                ">
                    Threshold
                </span>

                <span style="
                    position:absolute;
                    right:0;
                ">
                    Above threshold
                </span>

            </div>
        </div>

            """
        )
    )

    st.caption(
        f"{interpretation}. "
        f"Estimated risk: {probability:.1%} · "
        f"Research threshold: {threshold:.1%}."
    )
    
def prediction_card(
    prediction: Prediction,
) -> None:
    spec = MODEL_SPECS[prediction.model_id]
    display_probability = prediction.probability

    if prediction.model_id == "cprd":
        display_probability = min(
            prediction.probability * 3.5,
            1.0,
        )

    with st.container(border=True):
        st.markdown(
            f"#### {prediction.model_name}"
        )

        if not prediction.calibrated:
            st.error(
                "A validated calibrated cancer-risk "
                "estimate is not available for this model."
            )
            return

            if prediction.model_id in {
            "mimic",
            "eicu",
        }:
                st.caption(
                "Internally calibrated estimate of the "
                "modelled high-confidence silver phenotype. "
                "This is not a confirmed lung-cancer "
                "probability."
            )

        elif prediction.prediction_horizon_months:
            st.caption(
                "Estimated probability of the modelled "
                "lung-cancer outcome within "
                f"{prediction.prediction_horizon_months} "
                "months"
            )
        else:
            st.caption(
                "Estimated calibrated lung-cancer risk"
            )

        st.markdown(
            f"""
            <div class="risk-number">
                {risk_badge(prediction.category)}
                {display_probability:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"**{prediction.category}**"
        )

        threshold_label = (
            "Internal research operating threshold"
            if prediction.model_id in {
                "mimic",
                "eicu",
            }
            else "Selected investigation threshold"
        )

        st.caption(
            f"{threshold_label}: "
            f"{prediction.threshold:.1%} · "
            "Input coverage: "
            f"{prediction.input_coverage:.0%}"
        )

        render_risk_scale(
            probability=display_probability,
            threshold=prediction.threshold,
        )
        st.caption(spec.population)


def render_results(predictions: Mapping[str, Prediction], safety_flag: bool) -> None:
    st.subheader("Risk assessment result")
    if any(
        prediction.placeholder
        for prediction in predictions.values()
    ):
        result_notice = (
            "<strong>Prototype mode:</strong> One or more "
            "displayed results use a placeholder scoring "
            "function."
        )
    else:
        result_notice = (
            "<strong>Research-model mode:</strong> The "
            "displayed values come from exported research "
            "models with internal calibration. They are not "
            "clinically validated for patient-care decisions."
        )

    st.markdown(
        f'<div class="prototype">{result_notice}</div>',
        unsafe_allow_html=True,
    )
    if safety_flag:
        st.warning(
        "**Warning features remain active.** "
        "The prediction does not override clinical findings "
        "or an urgent referral pathway."
    )

    # This must not be indented inside the safety_flag condition.
    cols = st.columns(
        min(
            len(predictions),
            4,
        )
    )
    
    for col, prediction in zip(
        cols,
        predictions.values(),):
        with col:
            prediction_card(prediction)

    tabs = st.tabs(["Interpretation", "Model details"])
    primary = next(iter(predictions.values()))
    with tabs[0]:
        st.write(
            "The estimated risk should be considered together with the presenting complaint, examination, relevant "
            "guidance and patient preferences. It is not a diagnosis and does not automatically determine imaging or referral."
        )
        for warning in primary.warnings:
            st.warning(warning)
    with tabs[1]:
        for prediction in predictions.values():
            spec = MODEL_SPECS[prediction.model_id]
            with st.expander(spec.display_name):
                st.write(spec.description)
                st.write(f"**Population:** {spec.population}")
                if prediction.calibrated:
                    st.write(
                        "**Status:** Exported research model "
                        "with internal probability calibration."
                    )
                    st.write(
                        "**Calibration method:** "
                        f"{prediction.calibration_method}"
                    )
                    st.write(
                        "**Calibration population:** "
                        f"{prediction.calibration_population}"
                    )
                    st.write(
                        "**Research operating threshold:** "
                        f"{prediction.threshold:.2%}"
                    )
                    st.caption(
                        "This is not a clinically validated "
                        "referral or investigation threshold."
                    )
                else:
                    st.write(
                        "**Status:** Placeholder adapter "
                        "awaiting a trained pipeline and "
                        "calibration."
                    )


def collect_decision(safety_flag: bool) -> Dict[str, Any]:
    st.subheader("Clinical decision and safety-netting")
    st.caption("This records the clinician’s decision separately from the model output.")
    c1, c2 = st.columns(2)
    with c1:
        actions = st.multiselect(
            "Actions considered",
            [
                "Further clinical assessment",
                "Chest X-ray",
                "CT imaging",
                "Specialist referral",
                "Alternative diagnosis/investigation",
                "Safety-netting",
                "Planned review",
            ],
            default=["Safety-netting"] if not safety_flag else ["Further clinical assessment"],
        )
        urgency = st.selectbox("Clinical priority", ["Not recorded", "Routine", "Soon", "Urgent"])
    with c2:
        follow_up = st.text_input("Follow-up plan", placeholder="For example: review if symptoms persist or worsen")
        rationale = st.text_area("Clinical rationale", placeholder="Explain how clinical findings and the model output were considered")
    acknowledged = st.checkbox(
        "I understand that this prototype prediction does not replace clinical judgement or applicable guidance."
    )
    return {
        "actions_considered": actions,
        "clinical_priority": urgency,
        "follow_up_plan": follow_up,
        "clinical_rationale": rationale,
        "prototype_acknowledged": acknowledged,
    }


def serialise_prediction(prediction: Prediction) -> Dict[str, Any]:
    data = asdict(prediction)
    data["contributions"] = [{"feature": name, "relative_weight": value} for name, value in prediction.contributions]
    return data


def make_report(
    case_id: str,
    features: Mapping[str, Any],
    context: Mapping[str, Any],
    predictions: Mapping[str, Prediction],
    decision: Mapping[str, Any],
    safety_flag: bool,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_version": APP_VERSION,
        "prototype_notice": "All predictions currently use a non-clinical placeholder scoring function.",
        "patient_features": {FEATURE_LABELS.get(k, k): v for k, v in features.items()},
        "consultation_context": dict(context),
        "configured_warning_features_present": safety_flag,
        "model_results": [serialise_prediction(prediction) for prediction in predictions.values()],
        "clinical_decision": dict(decision),
        "disclaimer": (
            "Research demonstrator only. The output is not a diagnosis and must not be used to delay clinically "
            "appropriate investigation, referral or treatment."
        ),
    }


def sidebar() -> Tuple[str, List[str]]:
    with st.sidebar:
        st.markdown("### PHASE IV AI")
        st.caption(APP_SHORT_TITLE)
        st.markdown("**Assessment workspace**")
        mode = st.radio("View", ["Clinical demonstration", "Research comparison"])
        if mode == "Clinical demonstration":
            label_to_id = {
                spec.display_name: model_id
                for model_id, spec
                in MODEL_SPECS.items()
                if not spec.research_only
            }
            chosen_label = st.selectbox("Assessment model", list(label_to_id), index=0)
            selected = [label_to_id[chosen_label]]
        else:
            research_model_ids = list(
                MODEL_SPECS
            )

            selected = st.multiselect(
                "Models to compare",
                options=research_model_ids,
                default=research_model_ids,
                format_func=lambda model_id: (
                    MODEL_SPECS[
                        model_id
                    ].display_name
                ),
            )


        st.divider()
        
        st.markdown("### Project information")
        
        st.caption(
            "**PHASE IV AI**  \n"
            "Privacy compliant health data as a service "
            "for AI development"
        )
        
        st.caption(
            "**Grant agreement:** 101095384  \n"
            "**Programme:** Horizon Europe – Health  \n"
            "**Project period:** Oct 2023 – Dec 2026  \n"
            "**Coordinator:** University of Turku  \n"
            "**Demonstrator partner:** Nottingham Trent University"
        )
        
        st.markdown(
        "[PHASE IV AI website]"
        "(https://www.phase4ai-project.eu/)"
    )
    
        st.markdown(
        "[Official EU project record]"
        "(https://cordis.europa.eu/project/id/101095384)"
    )
            
        st.caption(f"{APP_VERSION}")
        return mode, selected


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🫁", layout="wide")
    inject_css()
    initialise_state()
    st.session_state.predictions = {}
    st.session_state.assessment_complete = False
    mode, selected_models = sidebar()

    render_project_header()

    if not selected_models:
        st.info("Select at least one model from the sidebar to continue.")
        return

    st.info(
        "The tool supports clinical assessment; it does not diagnose lung cancer or replace applicable referral guidance."
    )

    with st.form("patient_assessment", clear_on_submit=False):
        features, context = collect_assessment()
        submitted = st.form_submit_button("Review and estimate risk", type="primary", use_container_width=True)

    if submitted:
        errors, warnings = validate(features, context)
        for error in errors:
            st.error(error)
        for warning in warnings:
            st.warning(warning)
        if not errors:
            st.session_state.patient_features = features
            st.session_state.clinical_notes = context
            st.session_state.predictions = run_models(features, selected_models)
            st.session_state.assessment_complete = True

    if st.session_state.assessment_complete:
        current_ids = set(st.session_state.predictions)
        if current_ids != set(selected_models):
            st.session_state.predictions = run_models(st.session_state.patient_features, selected_models)
        safety_flag = render_safety_check(st.session_state.patient_features, st.session_state.clinical_notes)
        render_results(st.session_state.predictions, safety_flag)
        st.session_state.decision = collect_decision(safety_flag)

        report = make_report(
            st.session_state.case_id,
            st.session_state.patient_features,
            st.session_state.clinical_notes,
            st.session_state.predictions,
            st.session_state.decision,
            safety_flag,
        )
        st.divider()
        c1, c2 = st.columns([2, 1])
        c1.success("Assessment summary is ready. The downloaded file contains no direct patient identifier unless entered in free text.")
        c2.download_button(
            "Download assessment summary",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"{st.session_state.case_id}_assessment.json",
            mime="application/json",
            use_container_width=True,
        )

        
        render_project_footer()


if __name__ == "__main__":
    main()
