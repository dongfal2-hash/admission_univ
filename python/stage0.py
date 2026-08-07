"""
stage0.py
====================================================================
Shared low-level utilities — IDENTICAL in role to stage0.py used in the
Real Estate and Loan Eligibility projects. Reused as-is here because it
contains no project-specific logic:

- prepare result folders (txt / csv / visual)
- save a text report
- save a DataFrame to CSV
- save a matplotlib figure
- save a pickle / json object
- console + file logger

Project-specific logic (business question, feature engineering, model
training/selection, deployment/insight) is NOT implemented here — it
lives in each project's own stage1~4 modules.
"""

import os
import sys
import json
import pickle

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allow saving figures without a display
import matplotlib.pyplot as plt


def prepare_result_dirs(base_dir: str) -> dict:
    """Create txt / csv / visual subfolders under base_dir and return their paths."""
    paths = {
        "root": base_dir,
        "txt": os.path.join(base_dir, "txt"),
        "csv": os.path.join(base_dir, "csv"),
        "visual": os.path.join(base_dir, "visual"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


class ReportLogger:
    """Prints to console while also buffering the same lines to write to a text file."""

    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self._buffer = []

    def log(self, *args, sep=" ", end="\n"):
        line = sep.join(str(a) for a in args) + end
        sys.stdout.write(line)
        self._buffer.append(line)

    def section(self, title: str):
        bar = "=" * 70
        self.log("\n" + bar)
        self.log(title)
        self.log(bar)

    def save(self):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.writelines(self._buffer)


def save_fig(fig_or_plt, visual_dir: str, filename: str, dpi: int = 120):
    """Save the given figure (or the current plt figure) into visual_dir."""
    path = os.path.join(visual_dir, filename)
    if hasattr(fig_or_plt, "savefig"):
        fig_or_plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return path


def save_dataframe(df, csv_dir: str, filename: str):
    path = os.path.join(csv_dir, filename)
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path
