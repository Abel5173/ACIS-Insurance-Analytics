import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def run_ttest(group1, group2, group_var='Group', value_var='Value'):
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        return {
            'statistic': t_stat,
            'p_value': p_val,
            'test': f'T-test ({group_var})'
        }
    return {
        'statistic': np.nan,
        'p_value': np.nan,
        'test': f'T-test ({group_var})'
    }

def analyze_gender_claims(df, gender_col='Gender', value_col='TotalClaims', output_dir='../docs', plot_kind='box'):
    os.makedirs(output_dir, exist_ok=True)
    
    male_claims = df[df[gender_col] == 'Male'][value_col].dropna()
    female_claims = df[df[gender_col] == 'Female'][value_col].dropna()

    result = run_ttest(male_claims, female_claims, group_var='Male vs Female')

    # Save result to file
    with open(os.path.join(output_dir, 'task-3_results.txt'), 'a') as f:
        f.write(f"{result['test']}: statistic = {result['statistic']:.4f}, p-value = {result['p_value']:.4f}\n")

    # Plot
    plt.figure(figsize=(8, 6))
    if plot_kind == 'violin':
        sns.violinplot(data=df, x=gender_col, y=value_col, inner="box", scale="width")
    else:
        sns.boxplot(data=df, x=gender_col, y=value_col)
    plt.title(f"{value_col} by {gender_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{gender_col.lower()}_{value_col.lower()}_ttest_plot.png"))
    plt.show()
    plt.close()

    return result
