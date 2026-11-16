from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .utils import HPARAMS

CARE_TASK2_K_VALUES = tuple(HPARAMS["evaluation"]["ec_k_values"])
FULL_EC = re.compile(r"\d+\.\d+\.\d+\.\d+")  # same pattern as the CREEP wrapper


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


def compute_ec_accuracy_no_b(
    care_df: pd.DataFrame, *, k_values: tuple[int, ...] = CARE_TASK2_K_VALUES
) -> dict:
    """EC level accuracy without preliminary ECs (not four numeric levels).

    They leave the true ECs and the ranked lists. A reaction left without a true EC is not a query.
    Values are percent and unrounded, as in the CREEP and CLIPZyme wrappers.
    """
    ranks = rank_columns(care_df)
    queries = []
    for _, row in care_df.iterrows():
        true_ecs = [ec for ec in split_ec_labels(row["EC number"]) if FULL_EC.fullmatch(ec)]
        if true_ecs:
            ranked = [str(row[c]) for c in ranks if pd.notna(row[c]) and FULL_EC.fullmatch(str(row[c]))]
            queries.append((true_ecs, ranked))
    metrics = {}
    for k in k_values:
        scores = np.zeros((len(queries), 4))
        for i, (true_ecs, ranked) in enumerate(queries):
            levels = get_accuracy_level(ranked[:k], true_ecs)
            scores[i] = [np.mean([value >= level for value in levels]) for level in (1, 2, 3, 4)]
        means = scores.mean(axis=0) * 100.0
        metrics[f"k={k}"] = {f"level_{n}_accuracy": float(m) for n, m in zip((1, 2, 3, 4), means)}
    return {**metrics, "n_queries": len(queries), "b_ecs_dropped": True}
