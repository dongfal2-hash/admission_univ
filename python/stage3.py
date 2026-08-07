"""
stage3.py — Statistics & Machine Learning (Model Preparation, Training, Evaluation)
====================================================================
Key question: which model is best, and by what evaluation criteria?

Back to the SUPERVISED shape of the Loan project's stage3.py: a shared
train/test split, one independent train_model_*() function per model,
accuracy as the scoring criterion, compare_models() picking the winner.
What changed for this project:

  * The THING being compared is now "3 neural-network architectures"
    (was: 3 classifier families in the Loan project, 2 feature-set
    K-Means models in the clustering project):
      - MLP_relu     : hidden_layer_sizes=(3,),  activation=relu (default)
      - MLP_tanh     : hidden_layer_sizes=(3,),  activation=tanh
      - MLP_deep_8_4 : hidden_layer_sizes=(8,4), max_iter=800
    Each still gets its own train_model_*() function so a future
    architecture swap only touches one function.
  * TEST ACCURACY is the primary selection criterion (there IS a
    ground-truth label again), with the confusion matrix and per-class
    precision/recall logged for the selected view — silhouette/WCSS
    from the clustering project no longer apply.
  * Each MLP exposes loss_curve_ (training loss per iteration), which
    replaces the elbow/silhouette-vs-k scan as the "how did training
    behave" diagnostic — plot_loss_curves() draws all three.
  * cross_validate_models() (stratified K-Fold on the training split)
    is BACK as the stability check, replacing the clustering project's
    multi-seed silhouette check.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# Common input constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 123
BATCH_SIZE = 50
CV_FOLDS = 5


def _fit_and_score(name: str, estimator, xtrain_scaled, ytrain, xtest_scaled, ytest, log=None) -> dict:
    """
    Common helper used by every train_model_*(): fit -> predict -> score.
    Return-shape mirrors the other projects' _fit_and_score()
    (name/model/metrics) so downstream compare_models()/plot code stays
    structurally familiar.
    """
    estimator.fit(xtrain_scaled, ytrain)
    ypred_train = estimator.predict(xtrain_scaled)
    ypred_test = estimator.predict(xtest_scaled)

    metrics = {
        "model": name,
        "train_accuracy": accuracy_score(ytrain, ypred_train),
        "test_accuracy": accuracy_score(ytest, ypred_test),
    }
    cm = confusion_matrix(ytest, ypred_test)
    report = classification_report(ytest, ypred_test)

    if log:
        log.log(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        log.log(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        log.log(f"Confusion Matrix:\n{cm}")
        log.log(f"Classification Report:\n{report}")

    return {
        "name": name,
        "model": estimator,
        "ypred_test": ypred_test,
        "confusion_matrix": cm,
        "classification_report": report,
        "loss_curve": list(estimator.loss_curve_),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Model 1 — MLP with ReLU activation (default), single hidden layer of 3
# ---------------------------------------------------------------------------
MODEL1_NAME = "MLP_relu"


def train_model_relu(xtrain_scaled, ytrain, xtest_scaled, ytest, log=None) -> dict:
    """Model 1: MLPClassifier, hidden_layer_sizes=(3,), default relu activation."""
    if log:
        log.section(f"STAGE 3 — MODEL 1: MLPClassifier (activation=relu, default)")
    estimator = MLPClassifier(hidden_layer_sizes=(3,), batch_size=BATCH_SIZE,
                              max_iter=500, random_state=RANDOM_STATE)
    return _fit_and_score(MODEL1_NAME, estimator, xtrain_scaled, ytrain, xtest_scaled, ytest, log=log)


# ---------------------------------------------------------------------------
# Model 2 — MLP with tanh activation, single hidden layer of 3
# ---------------------------------------------------------------------------
MODEL2_NAME = "MLP_tanh"


def train_model_tanh(xtrain_scaled, ytrain, xtest_scaled, ytest, log=None) -> dict:
    """Model 2: MLPClassifier, hidden_layer_sizes=(3,), activation=tanh."""
    if log:
        log.section(f"STAGE 3 — MODEL 2: MLPClassifier (activation=tanh)")
    estimator = MLPClassifier(hidden_layer_sizes=(3,), batch_size=BATCH_SIZE,
                              max_iter=500, random_state=RANDOM_STATE, activation="tanh")
    return _fit_and_score(MODEL2_NAME, estimator, xtrain_scaled, ytrain, xtest_scaled, ytest, log=log)


# ---------------------------------------------------------------------------
# Model 3 — deeper MLP, hidden layers (8, 4)
# ---------------------------------------------------------------------------
MODEL3_NAME = "MLP_deep_8_4"


def train_model_deep(xtrain_scaled, ytrain, xtest_scaled, ytest, log=None) -> dict:
    """Model 3: MLPClassifier, hidden_layer_sizes=(8, 4), max_iter=800."""
    if log:
        log.section(f"STAGE 3 — MODEL 3: MLPClassifier (deeper, hidden_layer_sizes=(8,4))")
    estimator = MLPClassifier(hidden_layer_sizes=(8, 4), batch_size=BATCH_SIZE,
                              max_iter=800, random_state=RANDOM_STATE)
    return _fit_and_score(MODEL3_NAME, estimator, xtrain_scaled, ytrain, xtest_scaled, ytest, log=log)


# ---------------------------------------------------------------------------
# Stability check — stratified K-Fold CV on the training split
# ---------------------------------------------------------------------------
def cross_validate_models(model_results: list, xtrain_scaled, ytrain,
                          folds: int = None, log=None) -> pd.DataFrame:
    """K-Fold cross-validation of each architecture on the TRAINING split only —
    checks that the single 80/20 split didn't get lucky, without ever touching
    the held-out test set."""
    folds = folds if folds is not None else CV_FOLDS
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for r in model_results:
        # clone-by-params so CV re-fits fresh models rather than reusing fitted ones
        params = r["model"].get_params()
        estimator = MLPClassifier(**params)
        scores = cross_val_score(estimator, xtrain_scaled, ytrain, cv=skf, scoring="accuracy")
        rows.append({
            "model": r["name"],
            "cv_mean_accuracy": scores.mean(),
            "cv_std": scores.std(),
        })

    cv_df = pd.DataFrame(rows)
    if log:
        log.section(f"STAGE 3 — STABILITY CHECK ({folds}-fold stratified CV on the training split)")
        log.log(cv_df.round(4).to_string(index=False))
    return cv_df


# ---------------------------------------------------------------------------
# Comparing the architectures
# ---------------------------------------------------------------------------
def compare_models(model_results: list, log=None) -> dict:
    """
    Takes the results list from the train_model_*() functions, builds a
    comparison table, and picks the best model (highest TEST accuracy).
    """
    metrics_df = pd.DataFrame([r["metrics"] for r in model_results])
    metrics_df = metrics_df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["model"]

    trained_models = {r["name"]: r["model"] for r in model_results}
    cm_by_model = {r["name"]: r["confusion_matrix"] for r in model_results}
    loss_by_model = {r["name"]: r["loss_curve"] for r in model_results}
    pred_by_model = {r["name"]: r["ypred_test"] for r in model_results}

    if log:
        log.section("STAGE 3 — MODEL COMPARISON SUMMARY")
        log.log(metrics_df.round(4).to_string(index=False))
        log.log(f"\nBest model (highest test accuracy): {best_name}")

    return {
        "metrics": metrics_df,
        "trained_models": trained_models,
        "cm_by_model": cm_by_model,
        "loss_by_model": loss_by_model,
        "pred_by_model": pred_by_model,
        "best_model_name": best_name,
        "best_model": trained_models[best_name],
    }


def plot_model_comparison(metrics_df: pd.DataFrame, visual_dir: str, save_fig_fn,
                          filename: str = "accuracy_comparison.png", log=None) -> str:
    """Bar chart of test accuracy by architecture. Best model highlighted in green."""
    df_sorted = metrics_df.sort_values("test_accuracy", ascending=False)
    best_model = df_sorted.iloc[0]["model"]
    colors = ["seagreen" if m == best_model else "lightgray" for m in df_sorted["model"]]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(df_sorted["model"], df_sorted["test_accuracy"], color=colors)
    ax.set_title(f"Model Comparison — Best: {best_model}")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 3 — MODEL COMPARISON GRAPH")
        log.log(f"Saved: {path}")
        log.log(f"Best model highlighted: {best_model}")

    return path


def plot_loss_curves(loss_by_model: dict, visual_dir: str, save_fig_fn,
                     filename: str = "loss_curves.png", log=None) -> str:
    """Training loss per iteration for every architecture — the 'how did
    training behave' diagnostic unique to gradient-trained models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in loss_by_model.items():
        ax.plot(curve, label=name)
    ax.set_title("Training Loss Curve by Model")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 3 — LOSS CURVE GRAPH")
        log.log(f"Saved: {path}")

    return path


def data_modeling(
    stage2_out: dict,
    visual_dir: str = None,
    save_fig_fn=None,
    comparison_filename: str = "accuracy_comparison.png",
    loss_filename: str = "loss_curves.png",
    run_cv_check: bool = True,
    log=None,
) -> dict:
    """
    input: stage2_out — Stage 2 output dict with "xtrain_scaled"/"xtest_scaled"
           (the arrays the MLPs fit on) and "ytrain"/"ytest".

    order:
      1) train_model_relu() / train_model_tanh() / train_model_deep()
         — each independently fits and scores one architecture
      2) (optional) cross_validate_models() — K-Fold CV stability check
      3) compare_models()    — comparison table, best model picked (highest test accuracy)
      4) plot_loss_curves() / plot_model_comparison() — (optional) save PNGs + log

    output: {"metrics", "trained_models", "cm_by_model", "loss_by_model",
             "pred_by_model", "best_model_name", "best_model", "cv_metrics",
             "comparison_plot_path", "loss_plot_path"}
    """
    xtr, xte = stage2_out["xtrain_scaled"], stage2_out["xtest_scaled"]
    ytr, yte = stage2_out["ytrain"], stage2_out["ytest"]

    if log:
        log.section("STAGE 3 — HYPERPARAMETERS USED")
        log.log(f"batch_size={BATCH_SIZE}, random_state={RANDOM_STATE}; "
                f"architectures: relu(3,), tanh(3,), deep(8,4)")

    result_relu = train_model_relu(xtr, ytr, xte, yte, log=log)
    result_tanh = train_model_tanh(xtr, ytr, xte, yte, log=log)
    result_deep = train_model_deep(xtr, ytr, xte, yte, log=log)
    model_results = [result_relu, result_tanh, result_deep]

    cv_metrics = None
    if run_cv_check:
        cv_metrics = cross_validate_models(model_results, xtr, ytr, log=log)

    compared = compare_models(model_results, log=log)

    comparison_path = None
    loss_path = None
    if visual_dir and save_fig_fn:
        loss_path = plot_loss_curves(
            compared["loss_by_model"], visual_dir, save_fig_fn, filename=loss_filename, log=log,
        )
        comparison_path = plot_model_comparison(
            compared["metrics"], visual_dir, save_fig_fn, filename=comparison_filename, log=log,
        )

    return {
        **compared,
        "cv_metrics": cv_metrics,
        "comparison_plot_path": comparison_path,
        "loss_plot_path": loss_path,
    }
