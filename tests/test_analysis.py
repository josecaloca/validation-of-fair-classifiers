import pytest
import pandas as pd
import numpy as np

# Import your actual implementations instead of the dummy definitions used here
from utils.fairness_functions import (
    cdf,
    add_decile,
    agg_psi,
    PSI,
    score_percentile_comparison,
)


@pytest.fixture
def mock_dataframe() -> pd.DataFrame:
    """Fixture to create a standard test dataframe."""
    df = pd.DataFrame(
        {
            "Score": np.linspace(0.1, 0.9, 100),
            "ID": np.arange(100),
            "Sex": ["male"] * 50 + ["female"] * 50,
            "Protected": [0] * 60 + [1] * 40,
        }
    )
    df["Decile_rank"] = pd.qcut(df["Score"], 10, labels=False)
    return df


def test_cdf_returns_correct_values():
    sample = [1, 2, 2, 3, 3, 3]
    x, y = cdf(sample)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == 3
    assert len(y) == 3
    assert np.allclose(y, [1 / 6, 3 / 6, 1.0])


def test_add_decile_produces_expected_bins(mock_dataframe):
    result = add_decile(mock_dataframe, "Score")

    assert isinstance(result, pd.Series)
    assert result.min() == 0
    assert result.max() == 9
    assert result.notna().all()


def test_agg_psi_contains_required_columns(mock_dataframe):
    result = agg_psi(mock_dataframe, "Score", "ID")

    assert isinstance(result, pd.DataFrame)
    assert "perc" in result.columns
    assert np.isclose(result["perc"].sum(), 1.0)


def test_psi_computation_outputs_valid_psi(mock_dataframe):
    psi_df = PSI(mock_dataframe, mock_dataframe.copy(), "Score", "ID")

    assert isinstance(psi_df, pd.DataFrame)
    assert "PSI" in psi_df.columns
    assert psi_df["PSI"].notnull().all()
    assert (psi_df["PSI"] >= 0).all()


def test_score_percentile_comparison_returns_valid_structures(mock_dataframe):
    df_quantiles, df_regression = score_percentile_comparison(
        mock_dataframe,
        protected_variable="Protected",
        score="Score",
        favoured_class=0,
        deprived_class=1,
        plot=False,
    )

    assert isinstance(df_quantiles, pd.DataFrame)
    assert "difference" in df_quantiles.columns
    assert df_quantiles.shape[0] == 100

    assert isinstance(df_regression, pd.DataFrame)
    assert {"slope", "intercept", "p_value"}.issubset(df_regression.columns)
