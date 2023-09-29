"""
Author: Jose Caloca
Date: 21/04/2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, anderson_ksamp, ks_2samp, linregress
from sklearn.metrics import roc_curve, auc
import plotly.figure_factory as ff
import plotly.graph_objs as go


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve and fill the area under the curve.
    Args:
        fpr (array-like): False positive rate of ROC curve.
        tpr (array-like): True positive rate of ROC curve.
        roc_auc (float): AUC score of the ROC curve.

    Returns:
        None.
    """
    # Create a filled area under the ROC curve
    roc_trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc), fill='tozeroy')

    # Create a 45-degree diagonal line
    diagonal_trace = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Model', line=dict(color='black', dash='dash'))

    # Create a layout for the ROC plot
    layout = go.Layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        width=800,
        height=600,
        template="simple_white"
    )

    # Create a figure and add both traces
    fig = go.Figure(data=[roc_trace, diagonal_trace], layout=layout)

    # Show the ROC plot
    fig.show()


def plot_pd_hist(df_credit, threshold_dict=None):
    # Create a KDE plot for male and female data
    fig = ff.create_distplot([df_credit[df_credit['Sex'] == 'male']['Prob_default']], 
                            group_labels=['Male'],
                            colors=['blue'],
                            curve_type='kde',  # Set curve_type to 'kde' to fill the area
                            show_hist=False,
                            show_rug=False,
                            rug_text=None)

    # Add the KDE plot for females
    female_kde = ff.create_distplot([df_credit[df_credit['Sex'] == 'female']['Prob_default']], 
                                    group_labels=['Female'],
                                    colors=['red'],
                                    curve_type='kde',  # Set curve_type to 'kde' to fill the area
                                    show_hist=False,
                                    show_rug=False)
    fig.add_trace(female_kde.data[0])

    # Add vertical red dotted lines based on threshold_dict
    if threshold_dict is not None and isinstance(threshold_dict, dict):
        # Calculate the maximum density value for the KDE plot
        max_density = max(np.max(fig.data[0].y), np.max(female_kde.data[0].y))
        for label, threshold in threshold_dict.items():
            line_trace = go.Scatter(x=[threshold, threshold], 
                                    y=[0, max_density],  # Set Y-axis upper bound to max_density
                                    mode='lines', 
                                    name=label, 
                                    line=dict(dash='dash'))
            fig.add_trace(line_trace)

    # Update layout
    fig.update_layout(
        title='Probability of Default Distribution by Sex',
        xaxis_title='Probability of Default',
        yaxis_title='Density',
        template="simple_white",
        width=1000,
        height=600,
    )

    # Show the plot
    fig.show()
    
    
        
def cdf(sample):
    """Calculate the cumulative distribution function (CDF) of a sample.
    Args:
    sample (array-like): An array of numeric values.

    Returns:
        tuple: A tuple of two arrays (x, y), representing the x-axis and y-axis of the CDF. x contains the unique values in 
        the sample sorted in ascending order, and y contains the cumulative probabilities corresponding to x.
    """
    x, counts = np.unique(sample, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def add_decile(data, score_name):
    """Add a decile rank column to a dataframe based on a specified score.
    Args:
    data (pandas.DataFrame): A pandas dataframe containing the data.
    score_name (str): The name of the column containing the score to rank.

    Returns:
        pandas.Series: A pandas series containing the decile ranks for the specified score in the input dataframe.
    """
    return pd.qcut(data[score_name], 10, labels=False)


def agg_psi(sample, score_name, unique_key_name):
    """
    Aggregate data by decile rank and calculate the percentage of records in each decile.

    Args:
        sample (pandas.DataFrame): Input sample data.
        score_name (str): Name of the score column in the sample.
        unique_key_name (str): Name of the unique key column in the sample.

    Returns:
        pandas.DataFrame: Aggregated sample data with percentage of records in each decile.
    """
    sample = sample.groupby(["Decile_rank"]).agg(
        {score_name: ["mean", "min", "max"], unique_key_name: "count"}
    )
    sample["perc"] = sample[(unique_key_name, "count")] / sum(
        sample[(unique_key_name, "count")]
    )
    return sample


def PSI(sample1, sample2, score_name, unique_key_name):
    """
    Calculate the population stability index (PSI) between two samples.

    Args:
        sample1 (pandas.DataFrame): First sample for comparison.
        sample2 (pandas.DataFrame): Second sample for comparison.
        score_name (str): Name of the score column in the samples.
        unique_key_name (str): Name of the unique key column in the samples.

    Returns:
        pandas.DataFrame: Dataframe with PSI and decile rank columns.
    """
    samp1 = agg_psi(sample1, score_name, unique_key_name)

    samp2 = agg_psi(sample2, score_name, unique_key_name)

    mix = samp1.merge(samp2, how="left", on="Decile_rank")
    mix["PSI"] = (mix["perc_x"] - mix["perc_y"]) * np.log(mix["perc_x"] / mix["perc_y"])
    return mix



def score_percentile_comparison(
    df, protected_variable, score, favoured_class=0, deprived_class=1, plot=True
):
    """The objective of this test is to graphically compare the distributions of the scores assigned by the model 
    to the favoured and deprived sub-groups using a QQ-plot. Lastly, it fits a regression line of the QQ-plot to assess 
    the difference in the distributions based on the 80% rule.

    Args:
        df (pandas.DataFrame):      Dataset containing the model score and sensitive variable
        protected_variable (str):   String indicating the name of protected variable
        score (str):                String indicating the name of the variable containing the scores
        favoured_class (int):       Class of the favoured class in the protected variable
        deprived_class (int):       Class of the deprived class in the protected variable
        plot (bool):                Indicates whether to plot a QQ-plot. Default = True

    Returns:
        (pandas.DataFrame): Table summarising the percentiles and the difference

        (pandas.DataFrame): Table summarising the results of the linear regression
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
    if plot == True:
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
        plt.title(f"QQ plot between {favoured_class} (favoured) and {deprived_class} (deprived)")

    # Fit a regression line on the percentiles
    X = df_pct["q_deprived"]
    y = df_pct["q_favoured"]
    slope, intercept, r, p, se = linregress(y, X)
    # Get regression results
    linear_regression_result = pd.DataFrame(
        dict(slope=[slope], intercept=[intercept], p_value=[p])
    )

    return df_pct, linear_regression_result
