"""
stage1.py — Business Understanding & Data Collection
====================================================================
Key question: What business question is this project trying to answer,
and what shape is the raw data collected for it in?

Same 4-function shape as the Mall Customer Segmentation / Loan
Eligibility projects' stage1.py (business_overview -> data_overview ->
impute_missing -> clean_and_prepare), so the pipeline stays readable
across projects. What changed for this project:

  * This is a SUPERVISED BINARY CLASSIFICATION problem again (like the
    Loan project, unlike the clustering project) — but the raw target
    is CONTINUOUS: Admit_Chance is an estimated admission probability
    in [0, 1]. clean_and_prepare() therefore BINARIZES it at a 0.8
    threshold (>= 0.8 -> 1 "likely admitted", else 0), which replaces
    the Loan project's Y/N -> 1/0 target_map step.
  * Admission.csv can ship with a UTF-8 BOM / stray whitespace on the
    first column name, so clean_and_prepare() strips those from every
    column name before doing anything else (a data-collection quirk
    specific to this file).
  * Admission.csv has zero missing values, so impute_missing() is the
    same defensive/generic version used in the clustering project:
    median for numeric, mode for categorical, applied only to columns
    that actually contain nulls, logging "no missing values" otherwise.
  * The ID column here is Serial_No (was Customer_ID / Loan_ID).

Stage 2 (one-hot encoding / split / scaling / EDA) is not handled here.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# INPUT CONSTANTS — change these and the whole of Stage 1 follows
# ---------------------------------------------------------------------------
ID_COLUMN = "Serial_No"
TARGET_COLUMN = "Admit_Chance"
TARGET_THRESHOLD = 0.8   # Admit_Chance >= 0.8  ->  1 (likely admitted)

# Columns that are numerically coded but semantically categorical.
# They are cast to object dtype here so Stage 2's one-hot encoding
# picks them up (same intent as the Loan project recasting
# Credit_History / Loan_Amount_Term before encoding).
CATEGORICAL_CAST_COLS = ["University_Rating", "Research"]


def business_overview(log=None) -> str:
    """Defines the business question this project answers."""
    text = (
        "Business question: Can a graduate applicant's admission outcome be "
        "predicted from their academic profile (GRE, TOEFL, CGPA, SOP/LOR "
        "strength, university rating, research experience), and which neural-"
        "network architecture classifies 'likely admitted' (Admit_Chance >= 0.8) "
        "applicants most reliably?"
    )
    if log:
        log.section("STAGE 1 — BUSINESS UNDERSTANDING")
        log.log(text)
    return text


def data_overview(df: pd.DataFrame, log=None, n_head: int = 3, n_tail: int = 3) -> pd.DataFrame:
    """Logs shape / dtype / missing-value ratio / head / tail of the raw data."""
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "null_pct(%)": (df.isnull().mean() * 100).round(2),
    })

    if log:
        log.section("STAGE 1 — DATA COLLECTION: RAW OVERVIEW")
        log.log(f"Data shape: {df.shape}")
        log.log(f"\nColumn info:\n{info.to_string()}")
        log.log(f"\nColumn head:\n{df.head(n_head).to_string()}")
        log.log(f"\nColumn tail:\n{df.tail(n_tail).to_string()}")

    return info


def impute_missing(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """
    Generic missing-value handling: median for numeric columns, mode for
    categorical columns — but only applied to columns that actually have
    nulls. Admission.csv ships with zero missing values, so in the normal
    case this is a no-op that just confirms the data is clean.
    """
    before_nulls = df.isnull().sum()
    null_cols = before_nulls[before_nulls > 0].index.tolist()

    out = df.copy()
    applied = {}
    for col in null_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            value = out[col].median()
        else:
            value = out[col].mode()[0]
        out[col] = out[col].fillna(value)
        applied[col] = value

    if log:
        log.section("STAGE 1 — MISSING VALUE IMPUTATION")
        if null_cols:
            log.log(f"Missing values before imputation:\n{before_nulls.to_string()}")
            log.log(f"\nImputation strategy applied (auto median/mode): {applied}")
            log.log(f"\nMissing values after imputation:\n{out.isnull().sum().to_string()}")
        else:
            log.log("No missing values found — imputation skipped.")

    return out


def clean_and_prepare(
    df: pd.DataFrame,
    id_column: str = None,
    target_column: str = None,
    target_threshold: float = None,
    categorical_cast_cols: list = None,
    log=None,
) -> pd.DataFrame:
    """Strip BOM/whitespace from column names -> binarize the target ->
    drop the ID column -> cast numeric-coded categoricals to object dtype."""
    id_column = id_column if id_column is not None else ID_COLUMN
    target_column = target_column if target_column is not None else TARGET_COLUMN
    target_threshold = target_threshold if target_threshold is not None else TARGET_THRESHOLD
    categorical_cast_cols = (
        categorical_cast_cols if categorical_cast_cols is not None else CATEGORICAL_CAST_COLS
    )

    out = df.copy()

    # Admission.csv-specific quirk: the first column name can carry a BOM.
    out.columns = [c.strip().lstrip("\ufeff") for c in out.columns]

    # Binarize the continuous target (>= threshold -> 1).
    target_counts_raw = out[target_column].describe()
    out[target_column] = (out[target_column] >= target_threshold).astype(int)

    dropped = []
    if id_column and id_column in out.columns:
        out = out.drop(columns=[id_column])
        dropped.append(id_column)

    casted = [c for c in categorical_cast_cols if c in out.columns]
    for col in casted:
        out[col] = out[col].astype("object")

    if log:
        log.section("STAGE 1 — CLEAN & PREPARE")
        log.log(f"Raw {target_column} distribution (before binarization):\n{target_counts_raw.to_string()}")
        log.log(f"\nTarget binarized: {target_column} >= {target_threshold} -> 1, else 0")
        log.log(f"Binarized target counts:\n{out[target_column].value_counts().to_string()}")
        log.log(f"\nDropped columns: {dropped}")
        log.log(f"Cast to object dtype for one-hot encoding in Stage 2: {casted}")

    return out


def data_collection(
    raw_df: pd.DataFrame,
    id_column: str = None,
    target_column: str = None,
    target_threshold: float = None,
    categorical_cast_cols: list = None,
    save_dir: str = None,
    save_dataframe_fn=None,
    filename: str = "cleaned_df.csv",
    log=None,
) -> dict:
    """
    input: raw_df (Admission.csv raw DataFrame)

    order: business_overview -> data_overview(raw) -> impute_missing
           -> clean_and_prepare -> (optional) save CSV

    output: {"raw_df", "cleaned_df", "raw_info"}
    """
    business_overview(log=log)
    raw_info = data_overview(raw_df, log=log)
    imputed_df = impute_missing(raw_df, log=log)
    cleaned_df = clean_and_prepare(
        imputed_df,
        id_column=id_column,
        target_column=target_column,
        target_threshold=target_threshold,
        categorical_cast_cols=categorical_cast_cols,
        log=log,
    )

    if save_dir and save_dataframe_fn:
        save_dataframe_fn(cleaned_df, save_dir, filename)

    return {"raw_df": raw_df, "cleaned_df": cleaned_df, "raw_info": raw_info}
