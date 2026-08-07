"""
stage4.py — Model Deployment (Save Outputs, Visualize, Insight)
====================================================================
Key question: what insight can we get out of this model, and — in the
end — what should the business actually DO with it?

Takes Stage 3's output (trained_models / metrics / cm_by_model /
best_model / best_model_name) as input, exactly like the other
projects' stage4.py. Works for however many architectures were
compared — adding a 4th MLP variant needs no code changes here.
What changed for this project:

  * save_predictions() attaches predicted-vs-actual columns on the
    TEST split rows (there is an "actual" again) — the clustering
    project's cluster-label version is gone.
  * plot_confusion_matrices() is BACK (one heatmap per architecture),
    replacing the clustering project's plot_cluster_scatter().
  * plot_train_vs_test_accuracy() replaces plot_silhouette_comparison():
    same idea (bar chart comparing models) with the train/test PAIR
    plotted per model, because the overfitting gap — not a single
    score — is the key story for neural networks.
  * MLPs don't expose coefficients/importances the way Logistic
    Regression or Random Forest do, so generate_insight() profiles the
    correlation-with-target ranking from the cleaned data (computed in
    Stage 2's heatmap) plus the best model's per-class test performance,
    instead of a coefficient chart.
  * save_models() saves every trained architecture AND the fitted
    MinMaxScaler — the Streamlit "Predict Admission" tab must scale a
    new applicant's raw inputs the same way training data was scaled
    before calling .predict().
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_predictions(model_output: dict, stage2_out: dict, csv_dir: str, save_dataframe_fn,
                     filename: str = "test_predictions.csv") -> str:
    """Attach each architecture's test-set prediction (and the actual label)
    on the original (unscaled) test rows."""
    out_df = stage2_out["xtest"].copy()
    out_df["Actual"] = stage2_out["ytest"].values
    for name, pred in model_output["pred_by_model"].items():
        out_df[f"{name}_Pred"] = pred
    return save_dataframe_fn(out_df.reset_index(drop=True), csv_dir, filename)


def plot_confusion_matrices(model_output: dict, visual_dir: str, save_fig_fn,
                            filename: str = "confusion_matrices.png", log=None) -> str:
    """One confusion-matrix heatmap per architecture, side by side."""
    import seaborn as sns

    cm_by_model = model_output["cm_by_model"]
    names = list(cm_by_model.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        sns.heatmap(cm_by_model[name], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 4 — CONFUSION MATRICES")
        log.log(f"Saved: {path}")

    return path


def plot_train_vs_test_accuracy(model_output: dict, visual_dir: str, save_fig_fn,
                                filename: str = "train_vs_test_accuracy.png", log=None) -> str:
    """Train/test accuracy pair per architecture — makes the overfitting gap visible."""
    metrics_df = model_output["metrics"]
    x = range(len(metrics_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], metrics_df["train_accuracy"], width,
           label="train_accuracy", color="skyblue")
    ax.bar([i + width / 2 for i in x], metrics_df["test_accuracy"], width,
           label="test_accuracy", color="salmon")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy by Model")
    ax.legend()
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 4 — TRAIN VS TEST ACCURACY GRAPH")
        log.log(f"Saved: {path}")

    return path


def save_models(model_output: dict, scaler, root_dir: str, save_pickle_fn) -> dict:
    """Save every trained architecture AND the fitted scaler. Both are needed
    at inference time (Streamlit 'Predict Admission' tab must scale a new
    applicant's raw inputs before calling model.predict())."""
    paths = {}
    for name, model in model_output["trained_models"].items():
        path = save_pickle_fn(model, os.path.join(root_dir, f"{name}_Model.pkl"))
        paths[f"{name}_path"] = path

    scaler_path = save_pickle_fn(scaler, os.path.join(root_dir, "Admission_Scaler.pkl"))
    paths["scaler_path"] = scaler_path
    return paths


def generate_insight(model_output: dict, stage2_out: dict, log=None) -> str:
    """Business insight generated from the comparison table + the cleaned data's
    correlation-with-target ranking (MLPs expose no coefficient array, so the
    'which factors matter' answer comes from the data profile instead)."""
    metrics_df = model_output["metrics"]
    best_name = model_output["best_model_name"]
    best = metrics_df.set_index("model").loc[best_name]
    gap = best["train_accuracy"] - best["test_accuracy"]

    # correlation of each numeric input with the (binarized) target
    encoded_df = stage2_out["encoded_df"]
    target = stage2_out["ytrain"].name
    corr = encoded_df.corr()[target].drop(target).sort_values(ascending=False)
    top_factors = corr.head(3)

    lines = [
        f"[Model to deploy] {best_name} "
        f"(Test Accuracy {best['test_accuracy']:.3f}, Train Accuracy {best['train_accuracy']:.3f}, "
        f"train-test gap {gap:+.3f}).",
        "[Top factors correlated with admission]",
    ]
    for feat, r in top_factors.items():
        lines.append(f"  {feat}: r = {r:.3f}")

    lines.append(
        "[Action] Use the deployed model as a screening aid: surface each applicant's "
        "predicted admission likelihood next to their strongest profile factors "
        "(CGPA / GRE / TOEFL), route borderline predictions to human review, and "
        "advise applicants that raising the top-correlated factors moves their "
        "predicted outcome most."
    )

    insight = "\n".join(lines)
    if log:
        log.section("STAGE 4 — MODEL DEPLOYMENT / INSIGHT")
        log.log(insight)

    return insight


def model_deployment(
    model_output: dict,
    stage2_out: dict,
    csv_dir: str = None,
    save_dataframe_fn=None,
    visual_dir: str = None,
    save_fig_fn=None,
    root_dir: str = None,
    save_pickle_fn=None,
    log=None,
) -> dict:
    """
    input: model_output — Stage 3 data_modeling()'s return value
           stage2_out  — Stage 2's output (unscaled test rows, scaler, encoded table)

    order:
      1) save test-set predictions CSV          (csv_dir + save_dataframe_fn)
      2) confusion matrices + train/test chart  (visual_dir + save_fig_fn)
      3) save all trained models + scaler       (root_dir + save_pickle_fn)
      4) generate business insight + log

    Each step is silently skipped if its required arguments aren't given (partial runs OK).

    output: {"prediction_csv_path", "confusion_path", "train_test_path",
             "model_paths", "insight"}
    """
    result = {}

    if csv_dir and save_dataframe_fn:
        result["prediction_csv_path"] = save_predictions(
            model_output, stage2_out, csv_dir, save_dataframe_fn,
        )

    if visual_dir and save_fig_fn:
        result["confusion_path"] = plot_confusion_matrices(
            model_output, visual_dir, save_fig_fn, log=log,
        )
        result["train_test_path"] = plot_train_vs_test_accuracy(
            model_output, visual_dir, save_fig_fn, log=log,
        )

    if root_dir and save_pickle_fn:
        result["model_paths"] = save_models(
            model_output, stage2_out["scaler"], root_dir, save_pickle_fn,
        )

    result["insight"] = generate_insight(model_output, stage2_out, log=log)

    return result
