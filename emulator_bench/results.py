from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import HPARAMS

CARE_TASK2_K_VALUES = tuple(HPARAMS["evaluation"]["ec_k_values"])


def rank_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns if str(column).isdigit()]
    return sorted(columns, key=lambda value: int(value))


def split_ec_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    labels = []
    for raw in str(value).replace(",", ";").split(";"):
        label = raw.strip()
        if label and label.lower() not in {"nan", "none", "null", "-", "-.-.-.-"}:
            labels.append(label)
    return sorted(set(labels))


def get_accuracy_level(predicted_ecs: list[str], true_ecs: list[str]) -> list[int]:
    if not predicted_ecs:
        predicted_ecs = ["0.0.0.0"]
    levels = []
    for true_ec in true_ecs:
        true_parts = str(true_ec).split(".")
        counters = []
        for predicted_ec in predicted_ecs:
            predicted = str(predicted_ec)
            if predicted.count(".") != 3:
                predicted = "0.0.0.0"
            predicted_parts = predicted.split(".")
            counter = 0
            for predicted_part, true_part in zip(predicted_parts, true_parts):
                if predicted_part == true_part:
                    counter += 1
                else:
                    break
            counters.append(counter)
        levels.append(int(np.max(counters)) if counters else 0)
    return levels


def _average_accuracy(levels: list[int], level: int) -> float:
    if not levels:
        return 0.0
    return float(np.mean([1 if value >= level else 0 for value in levels]))


def compute_care_task2_metrics(
    care_df: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = CARE_TASK2_K_VALUES,
) -> dict:
    ranks = rank_columns(care_df)
    if not ranks:
        raise ValueError("CARE Task 2 DataFrame does not contain rank columns")
    metrics = {}
    for k in k_values:
        rows = []
        for _, row in care_df.iterrows():
            true_ecs = split_ec_labels(row["EC number"])
            predicted = [
                str(row[column])
                for column in ranks[:k]
                if pd.notna(row[column]) and str(row[column]).strip()
            ]
            rows.append(get_accuracy_level(predicted, true_ecs))
        metrics[f"k={k}"] = {
            f"level_{level}_accuracy": round(
                float(np.mean([_average_accuracy(levels, level) for levels in rows])) * 100.0,
                1,
            )
            for level in (4, 3, 2, 1)
        }
        metrics[f"k={k}"].update(
            {f"level_{level}_support": int(len(rows)) for level in (4, 3, 2, 1)}
        )
    return metrics
