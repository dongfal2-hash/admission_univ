"""
streamlit_app.py
====================================================================
UCLA Graduate Admission Prediction — 4-tab Streamlit app (unified dark
theme, same format as the Real Estate / Loan Eligibility / Mall
Customer Segmentation apps)

Principle: read only what already exists in results/ and slide/
           (pkl/csv/png). No retraining, no regenerating plots, and no
           rebuilding of the summary — cache the loads, compute only
           lightweight metrics on the fly (accuracy on the saved test
           predictions, a base-rate mean). Business/data logic (business
           question text, data overview, feature-row building) is
           imported from stage1.py / stage2.py rather than
           re-implemented here.

Layout on disk (see README.md): this file sits at the project root while
the stage modules live in python/, so python/ is prepended to sys.path
before importing them — this is what lets `streamlit run streamlit_app.py`
work from the project root (and on Streamlit Cloud, which always runs
from the repo root).

Tab order is the pipeline stages in REVERSE (Business Intelligence first, Dataset last),
so the accent colors are also reversed from the stage order (stage1 business needs=blue,
stage2 data engineering=purple, stage3 statistics/ML=green, stage4 BI intelligence=yellow):

Tab 1  Business Intelligence (yellow) : answer the business question using ONLY the
                                         single deployed/production model — enter an
                                         applicant's profile, get that model's
                                         admit-likely call + probability vs. the
                                         historical base rate (dynamic expectation).
                                         The relu/tanh/deep architecture comparison is
                                         NOT shown here — that's Tab 2's job.
Tab 2  Model Selection (green)        : why that model — train/test accuracy,
                                         CV stability, loss curves across the
                                         3 MLP architectures
Tab 3  EDA (purple)                    : which factors — data & feature engineering
                                         results (target distribution / correlation
                                         heatmap / feature distributions)
Tab 4  Dataset (blue)                   : which dataset — rows/columns, origin, raw
                                         overview, plus the pre-rendered project
                                         summary image (slide/summary.png)

Theme: black background / white text, set by the constants below. An optional
       .streamlit/config.toml can pin the same dark theme app-wide.
"""

import os
import pickle
import sys

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Paths — mirror the repository layout documented in README.md
#
#   ML04_admission_univ/
#   ├── data/Admission.csv
#   ├── python/stage0..4.py          <- imported below, hence the sys.path insert
#   ├── results/{*.pkl, csv/, txt/, visual/}
#   ├── slide/summary.png            <- pre-rendered summary shown in Tab 4
#   └── streamlit_app.py             <- this file
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(BASE_DIR, "python")
DATA_PATH = os.path.join(BASE_DIR, "data", "Admission.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR = os.path.join(RESULT_DIR, "csv")
VISUAL_DIR = os.path.join(RESULT_DIR, "visual")
SLIDE_DIR = os.path.join(BASE_DIR, "slide")
SUMMARY_PATH = os.path.join(SLIDE_DIR, "summary.png")

# The stage modules are in python/, not next to this file — make them importable
# regardless of the working directory the app is launched from.
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from stage1 import business_overview, data_overview, TARGET_THRESHOLD  # noqa: E402
from stage2 import build_feature_row                                   # noqa: E402

MODEL_NAMES = ["MLP_relu", "MLP_tanh", "MLP_deep_8_4"]

st.set_page_config(page_title="UCLA Admission Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Theme constants — shared across every tab (identical to the other apps)
# ---------------------------------------------------------------------------
COLOR_BLUE = "#8EC9F0"
COLOR_PURPLE = "#C9A6E8"
COLOR_GREEN = "#A8E6A3"
COLOR_YELLOW = "#FFF09E"

BG_COLOR = "#000000"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT_COLOR = "#B3B3B3"
FONT_FAMILY = "'Inter', sans-serif"

_SECTION_HEADER_TEMPLATE = """
<div style="border-left:4px solid {color}; padding:2px 0 2px 14px; margin-bottom:16px;">
  <div style="font-size:24px; font-weight:bold; color:{text}; font-family:{font};">{title}</div>
  {subtitle_html}
</div>
"""


def render_section_header(color: str, title: str, subtitle: str = None):
    """Consistent header style used at the top of every tab: colored left bar, no icons."""
    subtitle_html = (
        f'<div style="font-size:14px; color:{MUTED_TEXT_COLOR}; margin-top:4px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(_SECTION_HEADER_TEMPLATE.format(
        color=color, text=TEXT_COLOR, font=FONT_FAMILY, title=title, subtitle_html=subtitle_html,
    ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared image helper — every saved figure goes through here so a missing or
# not-yet-generated PNG shows a clear message instead of crashing the tab.
# ---------------------------------------------------------------------------
def show_saved_image(path: str, caption: str = None, **kwargs):
    if os.path.exists(path):
        st.image(path, caption=caption, **kwargs)
    else:
        st.warning(f"Missing figure: {os.path.relpath(path, BASE_DIR)} "
                   "— run the analysis pipeline in python/ to generate it.")


# ---------------------------------------------------------------------------
# Cached loaders — everything here only READS existing results (no retraining)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, "cleaned_df.csv"))


@st.cache_data
def load_encoded_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, "clean_admission_data.csv"))


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, "test_predictions.csv"))


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Stage 3's saved comparison table (model / train_accuracy / test_accuracy)."""
    df = pd.read_csv(os.path.join(CSV_DIR, "model_comparison.csv"))
    return df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)


@st.cache_resource
def load_model(name: str):
    with open(os.path.join(RESULT_DIR, f"{name}_Model.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(os.path.join(RESULT_DIR, "Admission_Scaler.pkl"), "rb") as f:
        return pickle.load(f)


def feature_columns() -> list:
    """Model input columns, in training order, straight from the saved encoded CSV."""
    encoded = load_encoded_data()
    return [c for c in encoded.columns if c != "Admit_Chance"]


# ---------------------------------------------------------------------------
# Tab 1 — Business Intelligence (predict a new applicant)
# ---------------------------------------------------------------------------
def render_business_intelligence_tab():
    metrics_df = load_model_metrics()
    best_name = metrics_df.iloc[0]["model"]

    render_section_header(
        COLOR_YELLOW,
        business_overview(),
        subtitle=f"Production model: {best_name} "
                 f"(test accuracy {metrics_df.iloc[0]['test_accuracy']:.3f})",
    )

    raw_df = load_raw_data()
    scaler = load_scaler()
    feat_cols = feature_columns()

    st.markdown(
        f"<div style='color:{TEXT_COLOR}; font-family:{FONT_FAMILY};'>"
        "Enter a new applicant's profile to predict their admission outcome:"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gre = st.slider("GRE Score", 260, 340, int(raw_df["GRE_Score"].median()))
        toefl = st.slider("TOEFL Score", 80, 120, int(raw_df["TOEFL_Score"].median()))
    with c2:
        sop = st.slider("SOP Strength", 1.0, 5.0, float(raw_df["SOP"].median()), 0.5)
        lor = st.slider("LOR Strength", 1.0, 5.0, float(raw_df["LOR"].median()), 0.5)
    with c3:
        cgpa = st.slider("CGPA (out of 10)", 6.0, 10.0, float(raw_df["CGPA"].median()), 0.01)
        rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
    with c4:
        research = st.radio("Research experience", ["Yes", "No"], horizontal=True)

    # One-hot shaped input dict; build_feature_row fills missing dummies with 0.
    inputs = {
        "GRE_Score": gre, "TOEFL_Score": toefl, "SOP": sop, "LOR": lor, "CGPA": cgpa,
        f"University_Rating_{rating}": 1,
        f"Research_{1 if research == 'Yes' else 0}": 1,
    }

    # LIVE prediction from the DEPLOYED model only — no button: Streamlit
    # reruns on every widget move, so the call updates in real time.
    # Tab 1 answers the business question ("what should we do for THIS
    # applicant, using the model we've chosen to deploy"), not "which
    # architecture is best" — that comparison lives in Tab 2.
    feature_row = build_feature_row(scaler, feat_cols, inputs)

    model = load_model(best_name)
    pred = int(model.predict(feature_row)[0])
    proba = float(model.predict_proba(feature_row)[0][1])
    label = "Likely admitted" if pred == 1 else "Unlikely"

    cleaned_df = load_cleaned_data()
    base_rate = cleaned_df["Admit_Chance"].mean()

    col_pred, col_gauge = st.columns([1, 2])
    with col_pred:
        st.metric(f"Prediction ({best_name})", label, f"P(admit) = {proba:.1%}")
    with col_gauge:
        # Dynamic expectation: this applicant's predicted probability vs the
        # historical base rate — the classification analogue of the
        # segment-vs-overall comparison in the clustering app.
        st.metric(
            "Vs. historical base rate",
            f"{base_rate:.1%} likely-admitted historically",
            f"{proba - base_rate:+.1%} for this applicant",
        )

    st.info(
        f"The deployed model ({best_name}, test accuracy "
        f"{metrics_df.iloc[0]['test_accuracy']:.3f}) estimates this applicant's admission "
        f"likelihood at {proba:.1%}, vs. a {base_rate:.1%} historical base rate "
        f"(Admit_Chance ≥ {TARGET_THRESHOLD}). This is a model estimate from one "
        f"historical snapshot — not an admissions decision."
    )


# ---------------------------------------------------------------------------
# Tab 2 — Model Selection (model efficiency / performance)
# ---------------------------------------------------------------------------
def render_model_selection_tab():
    render_section_header(COLOR_GREEN, "Model efficiency & performance",
                          subtitle="Comparison across 3 MLPClassifier architectures")

    metrics_df = load_model_metrics()
    best_name = metrics_df.iloc[0]["model"]

    def highlight_best(row):
        return [f"background-color: {COLOR_GREEN}; color: #000000" if row["model"] == best_name else ""
                for _ in row]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format({
            "train_accuracy": "{:.4f}", "test_accuracy": "{:.4f}",
        }),
        use_container_width=True,
    )

    # Lightweight verification from saved artifacts only: recompute each
    # architecture's test accuracy from the saved test predictions.
    preds = load_test_predictions()
    checks = {name: accuracy_score(preds["Actual"], preds[f"{name}_Pred"]) for name in MODEL_NAMES}
    st.success(
        f"Selected model: {best_name} (highest held-out test accuracy; verified from the "
        f"saved test predictions: "
        + ", ".join(f"{n} {a:.3f}" for n, a in checks.items()) + ")"
    )

    col1, col2 = st.columns(2)
    with col1:
        show_saved_image(os.path.join(VISUAL_DIR, "accuracy_comparison.png"),
                         caption="Test accuracy by architecture — best highlighted")
        show_saved_image(os.path.join(VISUAL_DIR, "train_vs_test_accuracy.png"),
                         caption="Train vs test accuracy — the overfitting-gap view")
    with col2:
        show_saved_image(os.path.join(VISUAL_DIR, "loss_curves.png"),
                         caption="Training loss per iteration — how each architecture converged")
        show_saved_image(os.path.join(VISUAL_DIR, "confusion_matrices.png"),
                         caption="Confusion matrices on the held-out test set")


# ---------------------------------------------------------------------------
# Tab 3 — EDA (data & feature engineering)
# ---------------------------------------------------------------------------
def render_eda_tab():
    render_section_header(COLOR_PURPLE, "EDA results", subtitle="Data engineering & feature engineering")

    col1, col2 = st.columns(2)
    with col1:
        show_saved_image(os.path.join(VISUAL_DIR, "target_distribution.png"),
                         caption=f"Binarized target — 1 if Admit_Chance ≥ {TARGET_THRESHOLD}")
    with col2:
        show_saved_image(os.path.join(VISUAL_DIR, "EDA_heatmap.png"),
                         caption="Feature correlation incl. the target — which factors track admission")

    show_saved_image(os.path.join(VISUAL_DIR, "EDA_feature_distributions.png"),
                     caption="Input feature distributions (GRE / TOEFL / SOP / LOR / CGPA)")
    st.caption(
        "MLPs are gradient-trained, so features were scaled to [0, 1] with MinMaxScaler "
        "(fit on the training split only — no test leakage) before fitting; otherwise "
        "GRE_Score (260–340) would dominate SOP/LOR (1–5) purely because of its numeric "
        "range. University_Rating and Research are numerically coded but semantically "
        "categorical, so they are one-hot encoded rather than treated as continuous."
    )


# ---------------------------------------------------------------------------
# Tab 4 — Dataset (rows/columns, origin)
# ---------------------------------------------------------------------------
def render_dataset_tab():
    render_section_header(COLOR_BLUE, "Dataset", subtitle="Admission.csv — graduate applicant records")

    raw_df = load_raw_data()
    cleaned_df = load_cleaned_data()
    encoded_df = load_encoded_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{raw_df.shape[0]:,}")
    c2.metric("Raw columns", raw_df.shape[1])
    c3.metric("Model input features (post-encoding)", encoded_df.shape[1] - 1)

    st.markdown(
        f"<div style='color:{TEXT_COLOR}; font-family:{FONT_FAMILY};'>"
        "Each row is one graduate applicant (Serial_No dropped after use), with GRE and "
        "TOEFL scores, SOP/LOR strength (1–5), undergraduate CGPA (out of 10), university "
        "rating (1–5), and research experience (0/1). The target Admit_Chance is an "
        f"estimated admission probability, binarized at {TARGET_THRESHOLD}: applicants at or "
        "above the threshold are labeled likely admitted (1), the rest unlikely (0)."
        "</div><br>",
        unsafe_allow_html=True,
    )

    info = data_overview(raw_df)
    st.dataframe(info, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download cleaned dataset (CSV)",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_df.csv",
            mime="text/csv",
        )
    with col2:
        # The test-prediction file IS this project's key deliverable —
        # predicted vs actual for every held-out applicant.
        preds = load_test_predictions()
        st.download_button(
            "Download test predictions (CSV)",
            data=preds.to_csv(index=False).encode("utf-8"),
            file_name="test_predictions.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------
    # Summary — the pre-rendered slide/summary.png, the same image the
    # README embeds. Nothing is composed in code here, so the app,
    # README and report all tell the identical story.
    # -----------------------------------------------------------------
    st.write("")
    render_section_header(COLOR_YELLOW, "Summary")
    show_saved_image(SUMMARY_PATH, use_container_width=True)

    st.caption("Tools: Python | pandas | scikit-learn | matplotlib | seaborn | Streamlit")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        f"<h1 style='color:{TEXT_COLOR}; font-family:{FONT_FAMILY};'>UCLA Graduate Admission Prediction</h1>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Business Intelligence", "Model Selection", "EDA", "Dataset"]
    )
    with tab1:
        render_business_intelligence_tab()
    with tab2:
        render_model_selection_tab()
    with tab3:
        render_eda_tab()
    with tab4:
        render_dataset_tab()


if __name__ == "__main__":
    main()
