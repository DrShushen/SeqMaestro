"""Smoke test for drop_columns_randomly."""

import pandas as pd

from igsr.method.igsr import drop_columns_randomly


def _make_df():
    return pd.DataFrame({
        "target": [1, 2, 3],
        "a": [10, 20, 30],
        "b": [40, 50, 60],
        "c": [70, 80, 90],
        "d": [100, 110, 120],
        "e": [130, 140, 150],
    })


def test_basic_drop():
    df = _make_df()
    result = drop_columns_randomly(df, n_keep=2, target_columns=["target"], random_seed=42)
    non_target = [c for c in result.columns if c != "target"]
    assert len(non_target) == 2
    assert "target" in result.columns
    # Original df should be unchanged
    assert len(df.columns) == 6


def test_always_keep():
    df = _make_df()
    result = drop_columns_randomly(
        df, n_keep=3, target_columns=["target"], random_seed=42, always_keep_columns=["a", "b"]
    )
    non_target = [c for c in result.columns if c != "target"]
    assert len(non_target) == 3
    assert "a" in result.columns
    assert "b" in result.columns
    assert "target" in result.columns


def test_always_keep_more_than_n_keep():
    """If always_keep has more columns than n_keep, we keep all forced columns."""
    df = _make_df()
    result = drop_columns_randomly(
        df, n_keep=2, target_columns=["target"], random_seed=42, always_keep_columns=["a", "b", "c"]
    )
    assert "a" in result.columns
    assert "b" in result.columns
    assert "c" in result.columns
    assert "target" in result.columns


def test_always_keep_nonexistent_column():
    """Non-existent always_keep columns are warned about and ignored."""
    df = _make_df()
    result = drop_columns_randomly(
        df, n_keep=2, target_columns=["target"], random_seed=42, always_keep_columns=["a", "nonexistent"]
    )
    assert "a" in result.columns
    assert "target" in result.columns
    non_target = [c for c in result.columns if c != "target"]
    assert len(non_target) == 2


def test_deterministic_with_seed():
    df = _make_df()
    r1 = drop_columns_randomly(df, n_keep=2, target_columns=["target"], random_seed=99)
    r2 = drop_columns_randomly(df, n_keep=2, target_columns=["target"], random_seed=99)
    assert list(r1.columns) == list(r2.columns)


if __name__ == "__main__":
    test_basic_drop()
    test_always_keep()
    test_always_keep_more_than_n_keep()
    test_always_keep_nonexistent_column()
    test_deterministic_with_seed()
    print("All smoke tests passed!")
