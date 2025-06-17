import numpy as np
import pandas as pd
import os
from scripts.clean_data import clean_category, clean_df

def load_clean_data(file_path='../data/raw/MachineLearningRating_v3.txt'):
    os.makedirs('../data/processed', exist_ok=True)
    df = pd.read_csv(file_path, sep='|', lineterminator='\n', on_bad_lines='warn')
    df['CleanCoverCategory'] = df['CoverCategory'].apply(clean_category)
    df = df.rename(columns={'TotalClaims\r': 'TotalClaims'})

    if 'LossRatio' not in df.columns:
        df['LossRatio'] = (df['TotalClaims'] / df['TotalPremium'] * 100).replace([np.inf, -np.inf], np.nan)
    df.to_csv('../data/processed/cleaned_data.csv', index=False)
    df = clean_df(df)

    return df[['TotalClaims', 'LossRatio', 'CalculatedPremiumPerTerm', 'SumInsured', 'Province', 'Gender', 'VehicleType', 'CleanCoverCategory', 'CoverCategory']].dropna()