import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import json
import os

def run_full_anova(df, group_col, value_col, result_dir='docs'):
    # Ensure directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Prepare data groups
    categories = df[group_col].dropna().unique()
    groups = [df[df[group_col] == cat][value_col].dropna().values for cat in categories]

    # Descriptive statistics
    print("\n📊 Descriptive Stats:")
    for cat, vals in zip(categories, groups):
        print(f"{cat}: n={len(vals)}, mean={np.mean(vals):.2f}")

    # Assumption 1: Normality (Shapiro-Wilk test)
    print("\n🔍 Normality Test (Shapiro-Wilk):")
    for cat, vals in zip(categories, groups):
        if len(vals) >= 3:  # Shapiro requires at least 3 values
            w, p = shapiro(vals)
            print(f"{cat}: W={w:.4f}, p-value={p:.4f}")

    # Assumption 2: Homogeneity of Variances (Levene’s Test)
    print("\n📏 Levene’s Test for Equal Variance:")
    stat_levene, p_levene = levene(*groups)
    print(f"Levene statistic = {stat_levene:.4f}, p-value = {p_levene:.4f}")

    # Run ANOVA
    stat_anova, p_anova = stats.f_oneway(*groups)
    print("\n🧪 ANOVA Result:")
    print(f"F-statistic = {stat_anova:.4f}, p-value = {p_anova:.4f}")

    result = {
        'test': 'ANOVA (LossRatio ~ CleanCoverCategory)',
        'group_column': group_col,
        'value_column': value_col,
        'statistic': stat_anova,
        'p_value': p_anova,
        'levene_p_value': p_levene
    }

    # Visualization
    plt.figure(figsize=(14, 8))
    sns.barplot(x=group_col, y=value_col, data=df)
    plt.title(f"{value_col} by {group_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../{result_dir}/barplot_{group_col}_{value_col}.png')
    plt.show()

    return result
