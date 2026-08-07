"""
streamlit_app.py
====================================================================
UCLA Graduate Admission Prediction — 4-tab Streamlit app (unified dark
theme, same format as the Real Estate / Loan Eligibility / Mall
Customer Segmentation apps)

Principle: read only what already exists in results/ (pkl/csv/png).
           No retraining, no regenerating plots — cache the loads,
           compute only lightweight metrics on the fly (accuracy on
           the saved test predictions, groupby means). Business/data
           logic (business question text, data overview, feature-row
           building) is imported from stage1.py / stage2.py rather
           than re-implemented here.

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
                                         overview, plus the same 4-step pipeline
                                         summary used in the other apps

Theme: black background / white text. Colors and dark theme are also set
       app-wide in config.toml.
"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score

from stage1 import business_overview, data_overview, TARGET_THRESHOLD
from stage2 import build_feature_row

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Admission.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR = os.path.join(RESULT_DIR, "csv")
VISUAL_DIR = os.path.join(RESULT_DIR, "visual")
SLIDE_DIR = os.path.join(RESULT_DIR, "slide")

MODEL_NAMES = ["MLP_relu", "MLP_tanh", "MLP_deep_8_4"]

st.set_page_config(page_title="UCLA Admission Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Theme constants — shared across every tab (identical to the other apps)
# ---------------------------------------------------------------------------
COLOR_BLUE = "#8EC9F0"
COLOR_PURPLE = "#C9A6E8"
COLOR_GREEN = "#A8E6A3"
COLOR_YELLOW = "#FFF09E"

STEP_COLORS = {1: COLOR_BLUE, 2: COLOR_PURPLE, 3: COLOR_GREEN, 4: COLOR_YELLOW}
_NUMBER_LABEL = {1: "1\ufe0f\u20e3", 2: "2\ufe0f\u20e3", 3: "3\ufe0f\u20e3", 4: "4\ufe0f\u20e3"}

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

_CARD_TEMPLATE = """
<div style="background:{bg}; color:{text}; border:2px solid {color};
            border-radius:8px; padding:16px; height:100%; font-family:{font};">
  <div style="font-size:26px; color:{color}; font-weight:bold;">{number}</div>
  <div style="font-weight:bold; font-size:18px; margin:6px 0 10px 0; color:{text};">{title}</div>
  <ul style="margin:0; padding-left:18px; font-size:14px; line-height:1.5; color:{text};">{bullets}</ul>
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


def _bullets_html(items: list) -> str:
    return "".join(f"<li>{i}</li>" for i in items)


def _render_step_card(step: int, title: str, bullets: list):
    """4-step pipeline summary card, identical style to the other apps' Summary tab."""
    st.markdown(_CARD_TEMPLATE.format(
        bg=BG_COLOR, text=TEXT_COLOR, font=FONT_FAMILY,
        color=STEP_COLORS[step], number=_NUMBER_LABEL[step],
        title=title, bullets=_bullets_html(bullets),
    ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders — everything here only READS existing results (no retraining)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
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

    col_pred, col_gauge = st.columns([1, 2])
    with col_pred:
        st.metric(f"Prediction ({best_name})", label, f"P(admit) = {proba:.1%}")
    with col_gauge:
        # Dynamic expectation: this applicant's predicted probability vs the
        # historical base rate — the classification analogue of the
        # segment-vs-overall comparison in the clustering app.
        cleaned_df = load_cleaned_data()
        base_rate = cleaned_df["Admit_Chance"].mean()
        delta = proba - base_rate
        st.metric(
            "Vs. historical base rate",
            f"{base_rate:.1%} likely-admitted historically",
            f"{delta:+.1%} for this applicant",
        )

    st.info(
        f"The deployed model ({best_name}, test accuracy "
        f"{metrics_df.iloc[0]['test_accuracy']:.3f}) estimates this applicant's admission "
        f"likelihood at {proba:.1%}, vs. a {base_rate:.1%} historical base rate "
        f"(Admit_Chance \u2265 {TARGET_THRESHOLD}). This is a model estimate from one "
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
        st.image(os.path.join(VISUAL_DIR, "accuracy_comparison.png"),
                 caption="Test accuracy by architecture — best highlighted")
        st.image(os.path.join(VISUAL_DIR, "train_vs_test_accuracy.png"),
                 caption="Train vs test accuracy — the overfitting-gap view")
    with col2:
        st.image(os.path.join(VISUAL_DIR, "loss_curves.png"),
                 caption="Training loss per iteration — how each architecture converged")
        st.image(os.path.join(VISUAL_DIR, "confusion_matrices.png"),
                 caption="Confusion matrices on the held-out test set")


# ---------------------------------------------------------------------------
# Tab 3 — EDA (data & feature engineering)
# ---------------------------------------------------------------------------
def render_eda_tab():
    render_section_header(COLOR_PURPLE, "EDA results", subtitle="Data engineering & feature engineering")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(VISUAL_DIR, "target_distribution.png"),
                 caption=f"Binarized target — 1 if Admit_Chance \u2265 {TARGET_THRESHOLD}")
    with col2:
        st.image(os.path.join(VISUAL_DIR, "EDA_heatmap.png"),
                 caption="Feature correlation incl. the target — which factors track admission")

    st.image(os.path.join(VISUAL_DIR, "EDA_feature_distributions.png"),
             caption="Input feature distributions (GRE / TOEFL / SOP / LOR / CGPA)")
    st.caption(
        "MLPs are gradient-trained, so features were scaled to [0, 1] with MinMaxScaler "
        "(fit on the training split only — no test leakage) before fitting; otherwise "
        "GRE_Score (260\u2013340) would dominate SOP/LOR (1\u20135) purely because of its numeric "
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
    metrics_df = load_model_metrics()
    best = metrics_df.iloc[0]
    best_name = best["model"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{raw_df.shape[0]:,}")
    c2.metric("Raw columns", raw_df.shape[1])
    c3.metric("Model input features (post-encoding)", encoded_df.shape[1] - 1)

    st.markdown(
        f"<div style='color:{TEXT_COLOR}; font-family:{FONT_FAMILY};'>"
        "Each row is one graduate applicant (Serial_No dropped after use), with GRE and "
        "TOEFL scores, SOP/LOR strength (1\u20135), undergraduate CGPA (out of 10), university "
        "rating (1\u20135), and research experience (0/1). The target Admit_Chance is an "
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
    # 4-step pipeline summary — a single designed slide image
    # (results/slide/summary.png) instead of generated HTML cards.
    # -----------------------------------------------------------------
    st.write("")
    render_section_header(COLOR_YELLOW, "Summary")

    summary_img_path = os.path.join(SLIDE_DIR, "summary.png")
    if os.path.exists(summary_img_path):
        st.image(summary_img_path, use_container_width=True)
    else:
        st.warning(
            f"Summary slide not found at results/slide/summary.png. "
            f"Add the image there to display it here."
        )

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
