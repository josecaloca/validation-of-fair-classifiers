import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, anderson_ksamp, ks_2samp, linregress
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots the ROC curve and fills the area under the curve with a color based on the AUC.
    """
    # Set the color of the fill area based on the AUC
    if roc_auc < 0.7:
        color = 'red'
    elif (roc_auc >= 0.7) & (roc_auc < 0.8):
        color = 'yellow'
    else:
        color = 'green'

    # Plot ROC curve and fill area under curve
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.fill_between(fpr, tpr, 0, color=color, alpha=0.5)

    # Add axis labels, limits, and title
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # Add legend
    plt.legend(loc="lower right")

    # Show the plot
    plt.show()


def cdf(sample):
    x, counts = np.unique(sample, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def add_decile(data, score_name):
    return pd.qcut(data[score_name], 10, labels=False)


def agg_psi(sample, score_name, unique_key_name):
    sample = sample.groupby(["Decile_rank"]).agg(
        {score_name: ["mean", "min", "max"], unique_key_name: "count"}
    )
    sample["perc"] = sample[(unique_key_name, "count")] / sum(
        sample[(unique_key_name, "count")]
    )
    return sample


def PSI(sample1, sample2, score_name, unique_key_name):

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
