"""
Author: Jose Caloca
Date: 21/04/2023

"""

from typing import Optional, Dict, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import plotly.figure_factory as ff
import plotly.graph_objs as go


def plot_roc_curve(
    fpr: Union[list[float], np.ndarray],
    tpr: Union[list[float], np.ndarray],
    roc_auc: float,
) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve with the area under the curve filled in.

    This function uses Plotly to visualize the performance of a binary classifier,
    plotting the true positive rate (TPR) against the false positive rate (FPR) for various thresholds.

    Args:
        fpr (list[float] or np.ndarray): False positive rates.
        tpr (list[float] or np.ndarray): True positive rates.
        roc_auc (float): Area Under the Curve (AUC) score.

    Returns:
        None
    """
    # Create a filled area under the ROC curve
    roc_trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="ROC Curve (AUC = {:.2f})".format(roc_auc),
        fill="tozeroy",
    )

    # Create a 45-degree diagonal line
    diagonal_trace = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Model",
        line=dict(color="black", dash="dash"),
    )

    # Create a layout for the ROC plot
    layout = go.Layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=800,
        height=600,
        template="simple_white",
    )

    # Create a figure and add both traces
    fig = go.Figure(data=[roc_trace, diagonal_trace], layout=layout)

    # Show the ROC plot
    fig.show()


def plot_pd_hist(
    df_credit: pd.DataFrame, threshold_dict: Optional[Dict[str, float]] = None
) -> None:
    """
    Plot the probability of default (PD) distributions by sex using KDE plots.

    This function visualizes the score distribution for male and female groups
    and optionally overlays vertical lines for specified thresholds.

    Args:
        df_credit (pd.DataFrame): DataFrame containing 'Sex' and 'Prob_default' columns.
        threshold_dict (dict[str, float], optional): Dictionary of threshold labels and values
            to be shown as vertical dashed lines. Defaults to None.

    Returns:
        None
    """
    # Create a KDE plot for male and female data
    fig = ff.create_distplot(
        [df_credit[df_credit["Sex"] == "male"]["Prob_default"]],
        group_labels=["Male"],
        colors=["blue"],
        curve_type="kde",  # Set curve_type to 'kde' to fill the area
        show_hist=False,
        show_rug=False,
        rug_text=None,
    )

    # Add the KDE plot for females
    female_kde = ff.create_distplot(
        [df_credit[df_credit["Sex"] == "female"]["Prob_default"]],
        group_labels=["Female"],
        colors=["red"],
        curve_type="kde",  # Set curve_type to 'kde' to fill the area
        show_hist=False,
        show_rug=False,
    )
    fig.add_trace(female_kde.data[0])

    # Add vertical red dotted lines based on threshold_dict
    if threshold_dict is not None and isinstance(threshold_dict, dict):
        # Calculate the maximum density value for the KDE plot
        max_density = max(np.max(fig.data[0].y), np.max(female_kde.data[0].y))
        for label, threshold in threshold_dict.items():
            line_trace = go.Scatter(
                x=[threshold, threshold],
                y=[0, max_density],  # Set Y-axis upper bound to max_density
                mode="lines",
                name=label,
                line=dict(dash="dash"),
            )
            fig.add_trace(line_trace)

    # Update layout
    fig.update_layout(
        title="Probability of Default Distribution by Sex",
        xaxis_title="Probability of Default",
        yaxis_title="Density",
        template="simple_white",
        width=1000,
        height=600,
    )

    # Show the plot
    fig.show()


def cdf(sample: Union[list[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the cumulative distribution function (CDF) for a given sample.

    Args:
        sample (list[float] or np.ndarray): Input array of numeric values.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Sorted unique values (x-axis).
            - Corresponding cumulative probabilities (y-axis).
    """
    x, counts = np.unique(sample, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def add_decile(data: pd.DataFrame, score_name: str) -> pd.Series:
    """
    Assign decile ranks to a score column in a DataFrame.

    Each observation is assigned a decile (0 to 9) based on its score value.

    Args:
        data (pd.DataFrame): Input DataFrame containing the score column.
        score_name (str): Name of the column containing scores to rank.

    Returns:
        pd.Series: Series of decile ranks (integer values from 0 to 9).
    """
    return pd.qcut(data[score_name], 10, labels=False)


def agg_psi(
    sample: pd.DataFrame, score_name: str, unique_key_name: str
) -> pd.DataFrame:
    """
    Aggregate a sample by decile rank and compute the proportion of records per decile.

    Args:
        sample (pd.DataFrame): Input data that must include a 'Decile_rank' column.
        score_name (str): Column name for score to compute summary statistics.
        unique_key_name (str): Column used to count the number of observations per decile.

    Returns:
        pd.DataFrame: Aggregated DataFrame with mean, min, max of scores,
                      count of unique keys, and proportion of records (`perc`) per decile.
    """
    sample = sample.groupby(["Decile_rank"]).agg(
        {score_name: ["mean", "min", "max"], unique_key_name: "count"}
    )
    sample["perc"] = sample[(unique_key_name, "count")] / sum(
        sample[(unique_key_name, "count")]
    )
    return sample


def PSI(
    sample1: pd.DataFrame, sample2: pd.DataFrame, score_name: str, unique_key_name: str
) -> pd.DataFrame:
    """
    Compute the Population Stability Index (PSI) between two datasets.

    PSI measures the shift in distribution of a score between two samples
    by comparing proportions across deciles.

    Args:
        sample1 (pd.DataFrame): Baseline sample with 'Decile_rank' already assigned.
        sample2 (pd.DataFrame): Comparison sample with same schema as sample1.
        score_name (str): Column name of the score variable.
        unique_key_name (str): Column used to compute counts for PSI calculation.

    Returns:
        pd.DataFrame: DataFrame including decile-level proportions and PSI contribution.
    """
    samp1 = agg_psi(sample1, score_name, unique_key_name)

    samp2 = agg_psi(sample2, score_name, unique_key_name)

    mix = samp1.merge(samp2, how="left", on="Decile_rank")
    mix["PSI"] = (mix["perc_x"] - mix["perc_y"]) * np.log(mix["perc_x"] / mix["perc_y"])
    return mix


def score_percentile_comparison(
    df: pd.DataFrame,
    protected_variable: str,
    score: str,
    favoured_class: int = 0,
    deprived_class: int = 1,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Graphically and statistically compare score distributions between protected subgroups.

    This function creates a QQ-plot comparing score quantiles between a favoured and a deprived class,
    and performs a linear regression on the quantiles to quantify distributional differences
    based on fairness rules like the 80% rule.

    Args:
        df (pd.DataFrame): Dataset containing scores and the protected attribute.
        protected_variable (str): Column name for the protected variable (e.g. gender, age group).
        score (str): Column name for the model score.
        favoured_class (int): Value in protected_variable denoting the advantaged group.
        deprived_class (int): Value in protected_variable denoting the disadvantaged group.
        plot (bool): Whether to display the QQ-plot. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of quantiles for favoured and deprived groups, with their differences.
            - DataFrame summarizing the linear regression fit (slope, intercept, p-value).
    """
    # Segment the scores of the favoured and deprived sub groups
    scores_deprived = df.loc[df[protected_variable] == deprived_class, score]
    scores_favoured = df.loc[df[protected_variable] == favoured_class, score]
    # calculate the percentiles of the score of each group
    df_pct = pd.DataFrame()
    df_pct["q_deprived"] = np.percentile(scores_deprived, range(100))
    df_pct["q_favoured"] = np.percentile(scores_favoured, range(100))
    # calculate the difference in the scores
    df_pct["difference"] = df_pct["q_favoured"] - df_pct["q_deprived"]
    # Plot a QQ-plot. Default option is True
    if plot is True:
        plt.figure(figsize=(8, 8))
        plt.scatter(x="q_favoured", y="q_deprived", data=df_pct, label="Actual fit")
        sns.lineplot(
            x="q_favoured",
            y="q_favoured",
            data=df_pct,
            color="r",
            label="Line of perfect fit",
        )
        plt.xlabel("Quantile of Prob. of default - favoured")
        plt.ylabel("Quantile of Prob. of default - deprived")
        plt.legend()
        plt.title(
            f"QQ plot between {favoured_class} (favoured) and {deprived_class} (deprived)"
        )

    # Fit a regression line on the percentiles
    X = df_pct["q_deprived"]
    y = df_pct["q_favoured"]
    slope, intercept, r, p, se = linregress(y, X)
    # Get regression results
    linear_regression_result = pd.DataFrame(
        dict(slope=[slope], intercept=[intercept], p_value=[p])
    )

    return df_pct, linear_regression_result
