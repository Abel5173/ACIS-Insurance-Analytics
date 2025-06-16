import numpy as np
import scipy.stats as stats

def run_anova(groups, group_var, value_var='LossRatio'):
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
        return {'statistic': f_stat, 'p_value': p_val, 'test': f'ANOVA ({group_var})'}
    return {'statistic': np.nan, 'p_value': np.nan, 'test': f'ANOVA ({group_var})'}