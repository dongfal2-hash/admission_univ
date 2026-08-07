"""
admission_analysis.py
====================================================================
UCLA GRADUATE ADMISSION PREDICTION — Stage 0~4 pipeline

Data: Admission.csv

Same 4-stage structure as customer_clustering_analysis.py /
loan_eligibility_analysis.py:
  Stage 1  Business Understanding & Data Collection
  Stage 2  Data Engineering (Encoding, Split, Normalization & EDA)
  Stage 3  Statistics & Machine Learning (model prep / training / evaluation)
  Stage 4  Model Deployment (outputs / visuals / insight)

This replaces the original monolithic admission_neural_network.py:
same data steps, same three MLPClassifier architectures, same outputs —
but split into per-stage modules so each stage answers its own key
question and any single piece can be swapped without touching the rest.
"""

import os
import pandas as pd

from stage0 import prepare_result_dirs, ReportLogger, save_fig, save_dataframe, save_pickle
import stage1
import stage2
import stage3
import stage4

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Admission.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# INPUT CONSTANTS — change these and the whole pipeline follows
# (leave alone to use each stage module's own defaults)
# ---------------------------------------------------------------------------
# Stage 1 — business understanding / cleaning
ID_COLUMN = "Serial_No"
TARGET_COLUMN = "Admit_Chance"
TARGET_THRESHOLD = 0.8
CATEGORICAL_CAST_COLS = ["University_Rating", "Research"]

# Stage 2 — data engineering (encoding + split + scaling)
ONE_HOT_COLS = ["University_Rating", "Research"]
TEST_SIZE = 0.2
RANDOM_STATE = 123
SCALE_METHOD = "minmax"   # "minmax" or "standard"

# Stage 3 — modeling
RUN_CV_CHECK = True


def main():
    paths = prepare_result_dirs(RESULT_DIR)
    log = ReportLogger(os.path.join(paths["txt"], "neural_network_report.txt"))

    log.section("UCLA ADMISSION PREDICTION (NEURAL NETWORK)")
    raw_df = pd.read_csv(DATA_PATH)

    # ---------------- STAGE 1 — Business Understanding & Data Collection ----------------
    stage1_out = stage1.data_collection(
        raw_df,
        id_column=ID_COLUMN,
        target_column=TARGET_COLUMN,
        target_threshold=TARGET_THRESHOLD,
        categorical_cast_cols=CATEGORICAL_CAST_COLS,
        save_dir=paths["csv"],
        save_dataframe_fn=save_dataframe,
        filename="cleaned_df.csv",
        log=log,
    )
    cleaned_df = stage1_out["cleaned_df"]

    # ---------------- STAGE 2 — Data Engineering (Encoding, Split, Normalization & EDA) ----------------
    stage2_out = stage2.data_processing(
        cleaned_df,
        target_column=TARGET_COLUMN,
        one_hot_cols=ONE_HOT_COLS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        method=SCALE_METHOD,
        save_dir=paths["csv"],
        save_dataframe_fn=save_dataframe,
        visual_dir=paths["visual"],
        save_fig_fn=save_fig,
        log=log,
    )

    # ---------------- STAGE 3 — Statistics & Machine Learning ----------------
    stage3_out = stage3.data_modeling(
        stage2_out,
        visual_dir=paths["visual"],
        save_fig_fn=save_fig,
        run_cv_check=RUN_CV_CHECK,
        log=log,
    )

    # save the comparison table as CSV (same deliverable as the original script)
    save_dataframe(stage3_out["metrics"], paths["csv"], "model_comparison.csv")

    # ---------------- STAGE 4 — Model Deployment ----------------
    stage4_out = stage4.model_deployment(
        stage3_out,
        stage2_out,
        csv_dir=paths["csv"],
        save_dataframe_fn=save_dataframe,
        visual_dir=paths["visual"],
        save_fig_fn=save_fig,
        root_dir=paths["root"],
        save_pickle_fn=save_pickle,
        log=log,
    )

    log.save()
    print(f"\n[DONE] Results saved under: {RESULT_DIR}")

    return {"stage1": stage1_out, "stage2": stage2_out, "stage3": stage3_out, "stage4": stage4_out}


if __name__ == "__main__":
    main()
