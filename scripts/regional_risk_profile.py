import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_anova(groups, group_var='Group'):
    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        return {
            'statistic': f_stat,
            'p_value': p_val,
            'test': f'ANOVA ({group_var})'
        }
    return {
        'statistic': np.nan,
        'p_value': np.nan,
        'test': f'ANOVA ({group_var})'
    }

def analyze_province_loss_ratio(df, group_col='Province', value_col='LossRatio', output_dir='../docs', plot_kind='box'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare groups for ANOVA
    provinces = df[group_col].dropna().unique()
    province_groups = [df[df[group_col] == prov][value_col].dropna().values for prov in provinces]

    # Run ANOVA
    result = run_anova(province_groups, group_var=group_col)

    # Save result to file
    with open(os.path.join(output_dir, 'task-3_results.txt'), 'a') as f:
        f.write(f"{result['test']}: statistic = {result['statistic']:.4f}, p-value = {result['p_value']:.4f}\n")

    # Plot
    plt.figure(figsize=(12, 6))
    if plot_kind == 'violin':
        sns.violinplot(data=df, x=group_col, y=value_col, inner="box", scale="width")
    else:
        sns.boxplot(data=df, x=group_col, y=value_col)
    plt.xticks(rotation=45)
    plt.title(f"{value_col} by {group_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{group_col.lower()}_{value_col.lower()}_anova_plot.png"))
    plt.show()
    plt.close()

    # Optional Tukey HSD if p < 0.05
    if result['p_value'] < 0.05:
        tukey = pairwise_tukeyhsd(endog=df[value_col].dropna(),
                                  groups=df[group_col][df[value_col].notna()],
                                  alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df.to_csv(os.path.join(output_dir, 'province_lossratio_tukey.csv'), index=False)
        result['tukey_result'] = tukey_df

    return result
