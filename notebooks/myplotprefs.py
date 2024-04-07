# My plot preferences
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Show side-by-side boxplots-histograms for specified numerical features of dataframe
def histogram_boxplot_df(data, features, highlight_indices=None):
    nrows = len(features)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows*3))
    if nrows == 1:
        ax = [ax]
    for i, col in enumerate(features):
        sns.boxplot(data=data, x=col, ax=ax[i][0], showmeans=True, color="skyblue")
        sns.histplot(data=data, x=col, kde=False, ax=ax[i][1], palette="Blues")
        ax[i][1].axvline(data[col].mean(), color="green", linestyle="--")
        ax[i][1].axvline(data[col].median(), color="black", linestyle="-")

        if highlight_indices is not None:
            for idx in highlight_indices:
                sample_value = data.loc[idx, col]
                ax[i][1].plot(sample_value, 0, "ro", markersize=8)
        ax[i][0].set_ylabel(col)
    fig.tight_layout()
    plt.show()


# Show scatterplots for specified numerical features of dataframe with a target
def scatter_df(data, num_features, target, highlight_indices=None, save_fig=False, fig_pathname=None):
    num_cols = 2 if len(num_features) > 1 else 1
    num_rows = (len(num_features) + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*5), facecolor="white")
    if len(num_features) == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    for i, feature in enumerate(num_features):
        ax = axs[i]
        sns.scatterplot(data=data, x=feature, y=data[target], ax=ax, alpha=0.3)
        corr_coef = data[[feature, target]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f"Corr. Coef.: {corr_coef:.2f}", transform=ax.transAxes, fontsize=14,
                    va="top", ha="left")
        if highlight_indices is not None:
            for idx in highlight_indices:
                highlight_data = data.iloc[idx]
                ax.scatter(x=highlight_data[feature], y=highlight_data[target], color="red", s=100)
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_pathname)
    plt.show()


# Show barplots for categorical features of dataframe
def barplot_df(data, perc=True):
    for feature in data.select_dtypes(include="object"):
        num_categories = data[feature].nunique()
        total = len(data[feature])
        figsize = (num_categories * 1.5, 6)
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(data=data, x=feature, palette="tab20", ax=ax)
        ax.set_xlabel(feature, fontsize=max(14, num_categories*1.5))
        ax.set_ylabel("Count", fontsize=max(14, num_categories*1.5))
        ax.tick_params(axis="both", which="major", labelsize=max(12, num_categories*1.2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        for p in ax.patches:
            if perc == True:
                label = "{:.1f}%".format(100*p.get_height()/total)
            else:
                label = p.get_height()
            x = p.get_x() + p.get_width() / 2  # Width of the plot
            y = p.get_height()  # Height of the plot
            ax.annotate(label, (x, y), ha = "center", va = "center", size = 15, xytext = (0, 5), textcoords = "offset points")
        plt.show()


# Show swarmplots for all categorical features of a dataframe with single numerical feature
def swarmplot_df(data, y_feature, highlight_indices=None):
    for feature in data.select_dtypes(include="object"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num_categories = data[feature].nunique()
            figsize = (num_categories * 1.5, 6)
            fig, ax = plt.subplots(figsize=figsize)

            # Get all unique categories for the current feature
            unique_categories = data[feature].unique()

            # Plot all data points
            sns.swarmplot(data=data, x=feature, y=data[y_feature], s=3, ax=ax, order=unique_categories)

            # Highlight specific indices
            if highlight_indices is not None:
                highlight_data = data.iloc[highlight_indices]
                sns.swarmplot(data=highlight_data, x=feature, y=highlight_data[y_feature],
                              s=10, ax=ax, color='red', marker='o', order=unique_categories)

            ax.set_xlabel(feature, fontsize=max(14, num_categories * 1.5))
            ax.set_ylabel(y_feature, fontsize=max(14, num_categories * 1.5))
            ax.tick_params(axis="both", which="major", labelsize=max(12, num_categories * 1.2))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

            plt.show()


# Show frequency table between two features
def freqplot(data, feature1, feature2):
    freq = pd.crosstab(data[feature1], data[feature2])
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(freq, annot=True, cmap="Blues", fmt=".0f", ax=ax)
    plt.show()


# Show boxplot between two features
def boxplot(data, feature_x, feature_y):
    fig, ax = plt.subplots(figsize=(14,8))
    sns.boxplot(data=data, x=feature_x, y=feature_y)
    plt.show()


# Show scatterplot of predicted versus actual values for a list of models and predictions
def pred_vs_actual(y, y_preds, y_pred_names):
    # Create a figure with side-by-side subplots
    num_subplots = len(y_preds)
    fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

    for i, (y_pred, name) in enumerate(zip(y_preds, y_pred_names)):
        # Plot the scatterplot for each y_pred
        axs[i].scatter(y, y_pred, alpha=0.3)  # Set alpha to 0.3 for transparency
        axs[i].set_xlabel("Actual y")
        axs[i].set_ylabel("Predicted y")
        axs[i].plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], "r--")
        axs[i].set_xlim(np.min([np.min(y), np.min(y_pred)]), np.max([np.max(y), np.max(y_pred)]))
        axs[i].set_ylim(np.min([np.min(y), np.min(y_pred)]), np.max([np.max(y), np.max(y_pred)]))
        axs[i].set_title(name)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


