import numpy as np
import scipy.stats as stats

def run_pearson_corr(var1, var2):
    if len(var1) > 1 and len(var2) > 1:
        corr, p_val = stats.pearsonr(var1, var2)
        return {'correlation': corr, 'p_value': p_val, 'test': 'Pearson Correlation'}
    return {'correlation': np.nan, 'p_value': np.nan, 'test': 'Pearson Correlation'}