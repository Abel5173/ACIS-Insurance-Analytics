import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
import os

def analyze_vehicle_make_claims(
    df,
    min_count=100,
    value_var='TotalClaims',
    group_var='VehicleType',
    top_n_plot=10,
    output_dir='../docs',
    plot_type='box'
):

    os.makedirs(output_dir, exist_ok=True)

    # Filter valid makes
    valid_makes = df[group_var].value_counts()
    valid_makes = valid_makes[valid_makes >= min_count].index.tolist()
    make_pairs = list(combinations(valid_makes, 2))

    # Run pairwise t-tests
    ttest_results = []
    for make1, make2 in make_pairs:
        group1 = df[df[group_var] == make1][value_var].dropna()
        group2 = df[df[group_var] == make2][value_var].dropna()

        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        ttest_results.append({
            'Make1': make1,
            'Make2': make2,
            'T-statistic': t_stat,
            'P-value': p_val
        })

    ttest_df = pd.DataFrame(ttest_results)

    # Multiple testing correction (Bonferroni)
    ttest_df['Adjusted P-value'] = ttest_df['P-value'] * len(ttest_df)
    ttest_df['Significant'] = ttest_df['Adjusted P-value'] < 0.05

    # Save results
    result_path = os.path.join(output_dir, 'ttest_vehiclemake_claims.csv')
    ttest_df.to_csv(result_path, index=False)

    # Plot distribution for top N makes
    top_makes = df[group_var].value_counts().head(top_n_plot).index.tolist()
    subset = df[df[group_var].isin(top_makes)]

    plt.figure(figsize=(14, 8))
    if plot_type == 'violin':
        sns.violinplot(data=subset, x=group_var, y=value_var, inner='box', scale='width')
    else:
        sns.boxplot(data=subset, x=group_var, y=value_var)
    plt.title(f'{value_var} Distribution by {group_var} (Top {top_n_plot})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{group_var.lower()}_{value_var.lower()}_distribution.png')
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    return ttest_df



{
 "cells": [
  
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}