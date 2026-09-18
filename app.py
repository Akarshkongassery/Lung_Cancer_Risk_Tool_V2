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
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import streamlit as st


APP_TITLE = "PHASE IV AI Lung Cancer Risk Assessment"
APP_SHORT_TITLE = "Lung Cancer Use-Case Demonstrator"
APP_VERSION = "PHASE IV AI prototype 2.1"
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


MODEL_SPECS: Dict[str, ModelSpec] = {
    "cprd": ModelSpec(
        "cprd", "CPRD Aurum", "UK primary-care records", 0.030, "#007C83",
        "Primary-care model slot. Replace with the validated CPRD pipeline.",
    ),
    "synpre": ModelSpec(
        "synpre", "SynPre-FL", "Federated multi-cohort research population", 0.030,
        "#3156A3", "Proposed synthetic-pretraining plus federated-learning model slot.",
    ),
    "mimic": ModelSpec(
        "mimic", "MIMIC", "Hospital/critical-care cohort", 0.030, "#7656A6",
        "Hospital-cohort comparator; not assumed to be validated for primary care.",
    ),
    "eicu": ModelSpec(
        "eicu", "eICU", "Intensive-care cohort", 0.030, "#B06820",
        "Critical-care comparator; substantial population mismatch may apply.",
    ),
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
    placeholder: bool = True


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


MODEL_REGISTRY = {key: PlaceholderRiskModel(spec) for key, spec in MODEL_SPECS.items()}


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
        c1, c2 = st.columns(2)
        family_lung = yes_no("Family history of lung cancer", "family_lung")
        family_cancer = yes_no("Family history of another cancer", "family_cancer")
        previous_cancer = c1.multiselect(
            "Previous malignancies",
            ["Thyroid", "Stomach", "Kidney", "Ovarian", "Leukaemia", "Myeloma", "Melanoma", "Head/neck", "Bladder", "Pancreatic"],
        )
        other_history = c2.text_area("Other relevant history", placeholder="Optional clinical context")

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
            "gender_1": sex == "Male",
            "smk_qt_final_2_2": smoking,
            "alc_units_day_7": alcohol_units_week,
            "bmifinal2": bmi if bmi_known else 25.0,
            "famhlg_1": family_lung or family_cancer,
            "familyhcancer_1": family_cancer,
            "thyroid_ca_1": "Thyroid" in previous_cancer,
            "stomach_ca_1": "Stomach" in previous_cancer,
            "kidney_ca_1": "Kidney" in previous_cancer,
            "copd_com": copd,
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
            "abnormal_examination": abnormal_exam,
            "abnormal_prior_imaging": concerning_imaging,
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
    if context.get("abnormal_prior_imaging"):
        flags.append("abnormal prior imaging")
    if context.get("abnormal_examination"):
        flags.append("abnormal respiratory examination")
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


def run_models(features: Mapping[str, Any], selected_ids: List[str]) -> Dict[str, Prediction]:
    return {model_id: MODEL_REGISTRY[model_id].predict(features) for model_id in selected_ids}


def prediction_card(prediction: Prediction) -> None:
    spec = MODEL_SPECS[prediction.model_id]

    with st.container(border=True):
        st.markdown(
            f"#### {prediction.model_name}"
        )

        st.markdown(
            f"""
            <div class="risk-number">
                {risk_badge(prediction.category)}
                {prediction.probability:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(f"**{prediction.category}**")

        st.caption(
            f"Demonstration threshold: "
            f"{prediction.threshold:.1%} · "
            f"Input coverage: "
            f"{prediction.input_coverage:.0%}"
        )

        st.progress(
            min(
                prediction.probability
                / max(prediction.threshold * 2.0, 0.01),
                1.0,
            )
        )

        st.caption(spec.population)


def render_results(predictions: Mapping[str, Prediction], safety_flag: bool) -> None:
    st.subheader("Risk assessment result")
    st.markdown(
        '<div class="prototype"><strong>Prototype mode:</strong> These values come from a placeholder scoring function, '
        'not the exported research models. They must not be used for patient care.</div>',
        unsafe_allow_html=True,
    )
    if safety_flag:
        st.markdown(
            '<div class="safety"><strong>Warning features remain active.</strong> The prediction does not override '
            'clinical findings or an urgent referral pathway.</div>',
            unsafe_allow_html=True,
        )

    cols = st.columns(min(len(predictions), 4))
    for col, prediction in zip(cols, predictions.values()):
        with col:
            prediction_card(prediction)

    tabs = st.tabs(["Interpretation", "Contributing information", "Model details"])
    primary = next(iter(predictions.values()))
    with tabs[0]:
        st.write(
            "The estimated risk should be considered together with the presenting complaint, examination, relevant "
            "guidance and patient preferences. It is not a diagnosis and does not automatically determine imaging or referral."
        )
        for warning in primary.warnings:
            st.warning(warning)
    with tabs[1]:
        if primary.contributions:
            st.write(f"Factors influencing the **{primary.model_name}** placeholder estimate:")
            max_value = max(value for _, value in primary.contributions) or 1.0
            for name, value in primary.contributions:
                st.write(name)
                st.progress(value / max_value)
            st.caption("These are model contributions, not proven causal effects.")
        else:
            st.info("No positive placeholder contributions were identified.")
    with tabs[2]:
        for prediction in predictions.values():
            spec = MODEL_SPECS[prediction.model_id]
            with st.expander(spec.display_name):
                st.write(spec.description)
                st.write(f"**Population:** {spec.population}")
                st.write("**Status:** Placeholder adapter awaiting the trained pipeline, calibrator and validated threshold.")


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
            label_to_id = {spec.display_name: model_id for model_id, spec in MODEL_SPECS.items()}
            chosen_label = st.selectbox("Assessment model", list(label_to_id), index=0)
            selected = [label_to_id[chosen_label]]
        else:
            selected = st.multiselect(
                "Models to compare",
                options=list(MODEL_SPECS),
                default=list(MODEL_SPECS),
                format_func=lambda model_id: MODEL_SPECS[model_id].display_name,
            )
        st.divider()
        st.caption(f"{APP_VERSION}")
        st.warning("Research prototype. Placeholder model active.")
    return mode, selected


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🫁", layout="wide")
    inject_css()
    initialise_state()
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

        with st.expander("Developer handover: replacing placeholder models"):
            st.code(
                """# Replace an entry in MODEL_REGISTRY with an adapter exposing:
#     predict(features: Mapping[str, Any]) -> Prediction
#
# The adapter should load and apply, in this order:
# 1. feature-name/order mapping
# 2. fitted imputer/encoder/scaler
# 3. exported model
# 4. fitted probability calibrator
# 5. validated operating threshold
# 6. explanation method
MODEL_REGISTRY[\"cprd\"] = CPRDModelAdapter(artifact_directory=\"models/cprd\")""",
                language="python",
            )

    render_project_footer()


if __name__ == "__main__":
    main()
