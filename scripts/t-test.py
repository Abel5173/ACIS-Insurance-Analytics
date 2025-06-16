import numpy as np
from scipy.stats import stats


def run_ttest(group1, group2, group_var, value_var='TotalClaims'):
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        return {'statistic': t_stat, 'p_value': p_val, 'test': f'T-test ({group_var})'}
    return {'statistic': np.nan, 'p_value': np.nan, 'test': f'T-test ({group_var})'}