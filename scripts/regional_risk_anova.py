import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def analyze_regional_risk_profile(
    df,
    group_var='Province',
    value_var='LossRatio',
    min_group_size=30,
    perform_tukey=True,
    output_dir='docs',
    plot_type='box'
):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    valid_groups = df[group_var].value_counts()
    valid_groups = valid_groups[valid_groups >= min_group_size].index.tolist()
    data = df[df[group_var].isin(valid_groups)]

    # Prepare for ANOVA
    grouped_data = [group[value_var].dropna().values for name, group in data.groupby(group_var)]
    group_labels = data[group_var].unique()

    if len(grouped_data) > 1:
        f_stat, p_val = f_oneway(*grouped_data)
    else:
        f_stat, p_val = np.nan, np.nan

    result_dict = {
        'test': f'ANOVA ({group_var})',
        'statistic': f_stat,
        'p_value': p_val,
    }

    # Save ANOVA result
    with open(os.path.join(output_dir, 'task-3_results.txt'), 'a') as f:
        f.write(f"{result_dict['test']}: statistic = {f_stat:.4f}, p-value = {p_val:.4f}\n")

    # Plot distribution
    plt.figure(figsize=(14, 8))
    if plot_type == 'violin':
        sns.violinplot(data=data, x=group_var, y=value_var, inner='box', scale='width')
    else:
        sns.lineplot(data=data, x=group_var, y=value_var)
    plt.title(f'{value_var} by {group_var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{group_var.lower()}_{value_var.lower()}_anova_plot.png')
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    # Optional: Tukey's HSD if ANOVA is significant
    if perform_tukey and p_val < 0.05:
        tukey_result = pairwise_tukeyhsd(
            endog=data[value_var],
            groups=data[group_var],
            alpha=0.05
        )
        tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
        tukey_df.to_csv(os.path.join(output_dir, 'tukey_hsd_province_lossratio.csv'), index=False)
        result_dict['tukey'] = tukey_df

    return result_dict
