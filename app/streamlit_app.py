from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DEFAULT_METRICS = {
    "test_macro_f1": 0.9305,
    "failure_precision": 0.8788,
    "failure_recall": 0.8529,
    "confusion_matrix": [[1924, 8], [10, 58]],
}

DEFAULT_THRESHOLD = {
    "average_precision": 0.8835,
    "best_threshold": 0.8494,
    "best_failure_f1": 0.8871,
    "default_macro_f1": 0.9305,
    "tuned_macro_f1": 0.9417,
}


st.set_page_config(
    page_title="Contextual Predictive Maintenance IoT",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else fallback
    except (OSError, json.JSONDecodeError):
        return fallback


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{helper}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_failure_risk(
    air_temp: float,
    process_temp: float,
    rpm: int,
    torque: float,
    tool_wear: int,
    load_factor: float,
    maintenance_gap: int,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.04

    temp_gap = process_temp - air_temp
    if temp_gap > 12:
        score += 0.16
        reasons.append("high process-to-air temperature gap")
    elif temp_gap > 8:
        score += 0.08

    if torque > 55:
        score += 0.20
        reasons.append("high torque load")
    elif torque > 45:
        score += 0.10

    if rpm < 1250:
        score += 0.16
        reasons.append("low rotational speed under load")
    elif rpm > 2300:
        score += 0.10
        reasons.append("unusually high rotational speed")

    if tool_wear > 210:
        score += 0.24
        reasons.append("tool wear close to replacement range")
    elif tool_wear > 150:
        score += 0.12

    if load_factor > 85:
        score += 0.14
        reasons.append("heavy production load")
    elif load_factor > 70:
        score += 0.06

    if maintenance_gap > 45:
        score += 0.14
        reasons.append("long gap since last maintenance")
    elif maintenance_gap > 25:
        score += 0.06

    score = min(score, 0.96)
    if not reasons:
        reasons.append("sensor values are inside the normal operating band")
    return score, reasons


def risk_label(score: float) -> tuple[str, str]:
    if score >= 0.68:
        return "High", "#ef4444"
    if score >= 0.35:
        return "Medium", "#f59e0b"
    return "Low", "#16a34a"


def show_report_image(filename: str, caption: str) -> None:
    path = REPORTS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Report image not found: {filename}")


def confusion_matrix_plot(matrix: list[list[int]]) -> None:
    values = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Healthy", "Failure"])
    ax.set_yticks([0, 1], labels=["Healthy", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (row, col), value in np.ndenumerate(values):
        ax.text(col, row, f"{value:,}", ha="center", va="center", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        [data-testid="stSidebar"] {
            background: #101827;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        .hero {
            padding: 30px 34px;
            border-radius: 8px;
            background: linear-gradient(135deg, #102033 0%, #16415f 48%, #0f766e 100%);
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        }
        .hero h1 {
            font-size: 2.35rem;
            margin: 0 0 10px 0;
            letter-spacing: 0;
        }
        .hero p {
            font-size: 1rem;
            max-width: 880px;
            margin: 0;
            color: rgba(255,255,255,0.88);
        }
        .metric-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 18px 18px 16px;
            background: white;
            min-height: 126px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .metric-card span {
            display: block;
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .metric-card strong {
            display: block;
            color: #0f172a;
            font-size: 2rem;
            line-height: 1.15;
            margin-top: 10px;
        }
        .metric-card small {
            color: #475569;
            display: block;
            margin-top: 8px;
        }
        .section-note {
            color: #475569;
            font-size: 0.96rem;
            margin-bottom: 14px;
        }
        .risk-box {
            border-radius: 8px;
            padding: 20px;
            color: white;
            margin-bottom: 12px;
        }
        .step-box {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 15px 16px;
            height: 100%;
        }
        .step-box b {
            color: #0f172a;
        }
        .step-box p {
            color: #475569;
            margin: 7px 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
metrics = load_json(REPORTS / "lightgbm_metrics.json", DEFAULT_METRICS)
thresholds = load_json(REPORTS / "threshold_metrics.json", DEFAULT_THRESHOLD)

with st.sidebar:
    st.title("Factory Input")
    st.caption("Change the sensor context and review the predicted maintenance risk.")
    air_temp = st.slider("Air temperature (C)", 20.0, 45.0, 30.2, 0.1)
    process_temp = st.slider("Process temperature (C)", 25.0, 60.0, 40.5, 0.1)
    rpm = st.slider("Rotational speed (RPM)", 900, 2800, 1510, 10)
    torque = st.slider("Torque (Nm)", 10.0, 80.0, 42.0, 0.5)
    tool_wear = st.slider("Tool wear (minutes)", 0, 260, 125, 1)
    load_factor = st.slider("Production load (%)", 20.0, 100.0, 68.0, 1.0)
    maintenance_gap = st.slider("Days since maintenance", 0, 90, 21, 1)

st.markdown(
    """
    <div class="hero">
        <h1>Contextual Predictive Maintenance IoT</h1>
        <p>Interactive machine-health dashboard for estimating equipment failure risk, explaining model performance, and turning sensor context into practical maintenance actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_risk, tab_metrics, tab_reports = st.tabs(
    ["Overview", "Failure Risk Demo", "Model Metrics", "Reports"]
)

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Model", "LightGBM", "Gradient boosting classifier")
    with col2:
        metric_card("Macro F1", pct(float(metrics.get("test_macro_f1", 0))), "Balanced test performance")
    with col3:
        metric_card("Failure Recall", pct(float(metrics.get("failure_recall", 0))), "Catches failure cases")
    with col4:
        metric_card("Best Threshold", f"{float(thresholds.get('best_threshold', 0)):.2f}", "Tuned for failure F1")

    st.subheader("What This Project Shows")
    st.markdown(
        "<p class='section-note'>A complete predictive-maintenance workflow: sensor data preparation, class-imbalance handling, LightGBM modeling, threshold tuning, robustness checks, and a deployable Streamlit interface.</p>",
        unsafe_allow_html=True,
    )

    step_cols = st.columns(4)
    steps = [
        ("1. Sensor Context", "Temperature, speed, torque, wear, load, and maintenance gap are treated as operational context."),
        ("2. Model Training", "A failure classifier is trained and evaluated with metrics that matter for rare failure detection."),
        ("3. Threshold Tuning", "The decision threshold is tuned so failure recall and precision are more useful in practice."),
        ("4. Maintenance Action", "The dashboard converts risk into clear recommended actions for operators."),
    ]
    for col, (title, body) in zip(step_cols, steps):
        with col:
            st.markdown(f"<div class='step-box'><b>{title}</b><p>{body}</p></div>", unsafe_allow_html=True)

with tab_risk:
    score, reasons = score_failure_risk(
        air_temp, process_temp, rpm, torque, tool_wear, load_factor, maintenance_gap
    )
    label, color = risk_label(score)
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            f"""
            <div class="risk-box" style="background:{color};">
                <div style="font-size:0.9rem;font-weight:700;text-transform:uppercase;opacity:.88;">Estimated Risk</div>
                <div style="font-size:3.2rem;font-weight:800;line-height:1;">{label}</div>
                <div style="font-size:1.35rem;margin-top:10px;">Failure probability: {pct(score)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(score)
        st.caption("This demo score is a transparent rule-based estimator for the UI. The project metrics come from the trained LightGBM report artifacts.")

    with right:
        st.subheader("Main Signals")
        for reason in reasons:
            st.write(f"- {reason}")
        if label == "High":
            st.error("Recommended action: schedule inspection soon, reduce load where possible, and check tool wear before the next production cycle.")
        elif label == "Medium":
            st.warning("Recommended action: monitor this asset closely and review maintenance availability during the next shift.")
        else:
            st.success("Recommended action: continue normal operation and keep routine maintenance cadence.")

    st.subheader("Current Sensor Snapshot")
    snapshot = pd.DataFrame(
        {
            "Feature": [
                "Air temperature",
                "Process temperature",
                "Rotational speed",
                "Torque",
                "Tool wear",
                "Production load",
                "Days since maintenance",
            ],
            "Value": [
                f"{air_temp:.1f} C",
                f"{process_temp:.1f} C",
                f"{rpm:,} RPM",
                f"{torque:.1f} Nm",
                f"{tool_wear} min",
                f"{load_factor:.0f}%",
                f"{maintenance_gap} days",
            ],
        }
    )
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

with tab_metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Macro F1", pct(float(metrics.get("test_macro_f1", 0))))
    with col2:
        st.metric("Failure Precision", pct(float(metrics.get("failure_precision", 0))))
    with col3:
        st.metric("Failure Recall", pct(float(metrics.get("failure_recall", 0))))

    left, right = st.columns([1, 1])
    with left:
        matrix = metrics.get("confusion_matrix", DEFAULT_METRICS["confusion_matrix"])
        confusion_matrix_plot(matrix)
    with right:
        st.subheader("Threshold Summary")
        threshold_table = pd.DataFrame(
            [
                {"Metric": "Average precision", "Value": pct(float(thresholds.get("average_precision", 0)))},
                {"Metric": "Best failure F1", "Value": pct(float(thresholds.get("best_failure_f1", 0)))},
                {"Metric": "Default macro F1", "Value": pct(float(thresholds.get("default_macro_f1", 0)))},
                {"Metric": "Tuned macro F1", "Value": pct(float(thresholds.get("tuned_macro_f1", 0)))},
                {"Metric": "Best threshold", "Value": f"{float(thresholds.get('best_threshold', 0)):.4f}"},
            ]
        )
        st.dataframe(threshold_table, use_container_width=True, hide_index=True)
        st.info("For predictive maintenance, recall matters because missed failures are expensive. Precision still matters because false alarms waste maintenance time.")

with tab_reports:
    report_tab1, report_tab2, report_tab3, report_tab4 = st.tabs(
        ["Distributions", "Precision Recall", "Robustness", "Class Balance"]
    )
    with report_tab1:
        show_report_image("sensor_distributions.png", "Sensor feature distributions")
    with report_tab2:
        show_report_image("precision_recall_curve.png", "Precision-recall behavior across thresholds")
    with report_tab3:
        show_report_image("noise_robustness.png", "Model behavior under noisy sensor readings")
    with report_tab4:
        show_report_image("class_distribution.png", "Healthy vs failure class distribution")

st.divider()
st.caption(
    "Built with Streamlit, Pandas, NumPy, Matplotlib, and saved model reports. Values are for portfolio demonstration and maintenance decision support, not automated machine control."
)
