"""EC level accuracy without preliminary ECs."""

import pandas as pd

from emulator_bench.results import compute_ec_accuracy_no_b


def frame(rows):
    return pd.DataFrame(
        [{"EC number": truth, **{str(i): ec for i, ec in enumerate(ranked)}} for truth, ranked in rows]
    )


def test_preliminary_ecs_leave_truth_ranking_and_queries():
    df = frame(
        [
            ("1.1.1.1;1.1.1.B2", ["1.1.1.B3", "1.1.2.1", "1.1.1.1"]),  # B-EC leaves truth and ranking
            ("2.3.4.B1", ["2.3.4.5", "2.3.4.6", "2.3.4.7"]),  # only a B-EC as truth: not a query
            ("3.4.5.6", ["3.4.5.B9", "3.4.5.6", "3.4.5.7"]),
        ]
    )
    out = compute_ec_accuracy_no_b(df, k_values=(1, 2))
    assert out["n_queries"] == 2 and out["b_ecs_dropped"] is True
    # k=1: query 1 top is 1.1.2.1 (level 2 match), query 3 top is 3.4.5.6 (exact)
    assert out["k=1"] == {
        "level_1_accuracy": 100.0,
        "level_2_accuracy": 100.0,
        "level_3_accuracy": 50.0,
        "level_4_accuracy": 50.0,
    }
    assert out["k=2"]["level_4_accuracy"] == 100.0
