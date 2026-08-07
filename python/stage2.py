"""
stage2.py — Data Engineering (Encoding, Split, Normalization & EDA)
====================================================================
Key question: which variables, in what form, should feed the model —
and which factors most influence the admission outcome?

Same 3-part shape as the other projects' stage2.py (build the model
inputs -> run EDA -> save), with the content back in SUPERVISED form
(closer to the Loan project than the clustering project):

  * one_hot_encode() is BACK: University_Rating (1-5) and Research
    (0/1) are numerically coded but semantically categorical, so they
    are one-hot encoded (Stage 1 already cast them to object dtype) —
    the same "recast then encode" move the Loan project applied to
    Credit_History / Loan_Amount_Term.
  * split_target() / split_train_test() are BACK: there is a target
    (binarized Admit_Chance), so an 80/20 stratified train/test split
    (random_state=123) guards evaluation, exactly as in the Loan
    project. The clustering project had neither.
  * scale_features() fits MinMaxScaler ON THE TRAINING SPLIT ONLY and
    then transforms both splits — scaling lives here (not Stage 3)
    but is leakage-safe, unlike the clustering project where scaling
    the full data was fine because there was nothing to leak.
    MLPs are gradient-trained, so unscaled inputs (GRE ~290-340 vs
    SOP ~1-5) would slow or destabilize convergence.
  * EDA is target-aware again: plot_target_distribution() shows the
    binarized class balance, plot_feature_distributions() shows each
    input's shape, and plot_correlation_heatmap() includes the target
    so "which factors correlate with admission" is answered visually.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# INPUT CONSTANTS — change these and the whole pipeline follows
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Admit_Chance"
ONE_HOT_COLS = ["University_Rating", "Research"]
TEST_SIZE = 0.2
RANDOM_STATE = 123
SCALE_METHOD = "minmax"   # "minmax" (0-1 range) or "standard" (mean=0, std=1)

NUMERIC_EDA_COLS = ["GRE_Score", "TOEFL_Score", "SOP", "LOR", "CGPA"]


def _make_scaler(method: str = None):
    method = method if method is not None else SCALE_METHOD
    if method == "standard":
        return StandardScaler()
    return MinMaxScaler()


def one_hot_encode(df: pd.DataFrame, one_hot_cols: list = None, log=None) -> pd.DataFrame:
    """One-hot encode the numerically coded categorical columns."""
    one_hot_cols = one_hot_cols if one_hot_cols is not None else ONE_HOT_COLS
    present = [c for c in one_hot_cols if c in df.columns]
    encoded = pd.get_dummies(df, columns=present, dtype="int")

    if log:
        log.section("STAGE 2 — DATA ENGINEERING: ONE-HOT ENCODING")
        log.log(f"One-hot encoded columns: {present}")
        log.log(f"Shape before: {df.shape} -> after: {encoded.shape}")

    return encoded


def split_target(df: pd.DataFrame, target_column: str = None):
    """Split the encoded table into features X and target y."""
    target_column = target_column if target_column is not None else TARGET_COLUMN
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None,
    log=None,
):
    """80/20 stratified split so both splits keep the same class balance."""
    test_size = test_size if test_size is not None else TEST_SIZE
    random_state = random_state if random_state is not None else RANDOM_STATE

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if log:
        log.section("STAGE 2 — TRAIN/TEST SPLIT")
        log.log(f"Split: {int((1 - test_size) * 100)}/{int(test_size * 100)}, "
                f"stratified on target, random_state={random_state}")
        log.log(f"Train shape: {xtrain.shape}, Test shape: {xtest.shape}")
        log.log(f"Train target balance:\n{ytrain.value_counts(normalize=True).round(3).to_string()}")
        log.log(f"Test target balance:\n{ytest.value_counts(normalize=True).round(3).to_string()}")

    return xtrain, xtest, ytrain, ytest


def scale_features(xtrain: pd.DataFrame, xtest: pd.DataFrame, method: str = None, log=None):
    """Fit the scaler on the TRAINING split only, then transform both splits
    (leakage-safe). Returns (xtrain_scaled, xtest_scaled, fitted_scaler)."""
    scaler = _make_scaler(method)
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    if log:
        log.section("STAGE 2 — FEATURE SCALING")
        log.log(f"Scaler: {type(scaler).__name__} (fit on train only — no test leakage)")

    return xtrain_scaled, xtest_scaled, scaler


def build_feature_row(scaler, feature_cols: list, inputs: dict) -> pd.DataFrame:
    """
    Turn a single new applicant's raw inputs (dict) into a 1-row, correctly
    ordered, scaled DataFrame the model can predict on. Used by the
    Streamlit "Predict Admission" tab. inputs must already be one-hot
    shaped (same columns as the encoded training features).
    """
    row = pd.DataFrame([{col: inputs.get(col, 0) for col in feature_cols}])
    return scaler.transform(row)


# ---------------------------------------------------------------------------
# EDA — target distribution / feature distributions / correlation heatmap
# ---------------------------------------------------------------------------
def plot_target_distribution(
    df: pd.DataFrame,
    visual_dir: str,
    save_fig_fn,
    target_column: str = None,
    filename: str = "target_distribution.png",
    log=None,
) -> str:
    """Bar chart of the binarized target — the class balance the models must handle."""
    target_column = target_column if target_column is not None else TARGET_COLUMN
    counts = df[target_column].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Unlikely (0)", "Likely (1)"], counts.values, color=["salmon", "seagreen"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(f"Admission Outcome Distribution (1 = Admit_Chance >= 0.8)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 2 — EDA: TARGET DISTRIBUTION")
        log.log(f"Target counts:\n{counts.to_string()}")
        log.log(f"Positive-class share: {counts.get(1, 0) / counts.sum():.1%}")
        log.log(f"Saved: {path}")

    return path


def plot_feature_distributions(
    df: pd.DataFrame,
    visual_dir: str,
    save_fig_fn,
    numeric_cols: list = None,
    filename: str = "EDA_feature_distributions.png",
    log=None,
) -> str:
    """Histogram of every numeric input feature — the shape of the model's raw inputs."""
    numeric_cols = numeric_cols if numeric_cols is not None else NUMERIC_EDA_COLS
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=15, color="seagreen", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 2 — EDA: FEATURE DISTRIBUTIONS")
        for col in numeric_cols:
            log.log(f"{col}: mean={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}")
        log.log(f"Saved: {path}")

    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    visual_dir: str,
    save_fig_fn,
    filename: str = "EDA_heatmap.png",
    log=None,
) -> str:
    """Correlation heatmap among numeric features INCLUDING the binarized target,
    so 'which factors correlate with admission' is answered directly."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(6, 0.95 * n), max(5, 0.95 * n)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.columns)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap (incl. target)")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 2 — EDA: CORRELATION HEATMAP")
        target_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False) \
            if TARGET_COLUMN in corr.columns else None
        log.log(f"Correlation matrix:\n{corr.round(2).to_string()}")
        if target_corr is not None:
            log.log(f"\nCorrelation with target (sorted):\n{target_corr.round(3).to_string()}")
        log.log(f"Saved: {path}")

    return path


def data_processing(
    df: pd.DataFrame,
    target_column: str = None,
    one_hot_cols: list = None,
    test_size: float = None,
    random_state: int = None,
    method: str = None,
    save_dir: str = None,
    save_dataframe_fn=None,
    visual_dir: str = None,
    save_fig_fn=None,
    log=None,
) -> dict:
    """
    input: df (Stage 1 output cleaned_df — target already binarized)

    order: (optional) EDA plots on the cleaned table -> one_hot_encode
           -> split_target -> split_train_test -> scale_features
           -> (optional) save encoded CSV

    output: {"encoded_df", "feature_cols", "xtrain", "xtest", "ytrain", "ytest",
             "xtrain_scaled", "xtest_scaled", "scaler",
             "target_plot_path", "distribution_path", "heatmap_path"}
    """
    target_plot_path = None
    distribution_path = None
    heatmap_path = None
    if visual_dir and save_fig_fn:
        target_plot_path = plot_target_distribution(df, visual_dir, save_fig_fn,
                                                    target_column=target_column, log=log)
        distribution_path = plot_feature_distributions(df, visual_dir, save_fig_fn, log=log)
        heatmap_path = plot_correlation_heatmap(df, visual_dir, save_fig_fn, log=log)

    encoded_df = one_hot_encode(df, one_hot_cols=one_hot_cols, log=log)
    x, y = split_target(encoded_df, target_column=target_column)
    xtrain, xtest, ytrain, ytest = split_train_test(
        x, y, test_size=test_size, random_state=random_state, log=log,
    )
    xtrain_scaled, xtest_scaled, scaler = scale_features(xtrain, xtest, method=method, log=log)

    if save_dir and save_dataframe_fn:
        save_dataframe_fn(encoded_df, save_dir, "clean_admission_data.csv")

    return {
        "encoded_df": encoded_df,
        "feature_cols": list(x.columns),
        "xtrain": xtrain, "xtest": xtest, "ytrain": ytrain, "ytest": ytest,
        "xtrain_scaled": xtrain_scaled, "xtest_scaled": xtest_scaled,
        "scaler": scaler,
        "target_plot_path": target_plot_path,
        "distribution_path": distribution_path,
        "heatmap_path": heatmap_path,
    }
