import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def analyze_coverage_vehicle_interaction(df, cover_col='CleanCoverCategory', vehicle_col='VehicleType', 
                                         value_col='LossRatio', output_dir='../docs', plot=False):
    os.makedirs(output_dir, exist_ok=True)

    formula = f'{value_col} ~ C({cover_col}) * C({vehicle_col})'
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    interaction_term = f'C({cover_col}):C({vehicle_col})'
    interaction_p = anova_table['PR(>F)'][interaction_term]
    result = {
        'test': 'Two-Way ANOVA (Coverage × VehicleType)',
        'p_value': interaction_p
    }

    # Save result
    with open(os.path.join(output_dir, 'task-3_results.txt'), 'a') as f:
        f.write(f"{result['test']}: p-value = {interaction_p:.4f}\n")

    # Optional plot
    if plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x=cover_col, y=value_col, hue=vehicle_col, ci='sd')
        plt.title(f'{value_col} by {cover_col} and {vehicle_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'twoway_anova_{cover_col.lower()}_{vehicle_col.lower()}.png'))
        plt.show()
        plt.close()

    return result
