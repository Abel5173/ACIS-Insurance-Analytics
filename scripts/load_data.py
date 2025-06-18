import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.clean_data import clean_category, clean_df, clean_missing_values

def load_raw_data(file_path='../data/raw/MachineLearningRating_v3.txt'):
    os.makedirs('../data/processed', exist_ok=True)
    df = pd.read_csv(
        file_path,
        sep='|',
        lineterminator='\n',
        on_bad_lines='warn',
        low_memory=False
    )
    df = df.rename(columns={'TotalClaims\r': 'TotalClaims'})
    return df

def preprocess_data(df, drop_leaky_columns=True):
    print(f"Initial row count: {len(df)}")
    print("Pre-processing TotalClaims stats:\\n", df['TotalClaims'].describe())
    df['CleanCoverCategory'] = df['CoverCategory'].apply(clean_category)
    df['LossRatio'] = (df['TotalClaims'] / df['TotalPremium'] * 100).replace([np.inf, -np.inf], np.nan)
    df = clean_df(df)

    df = clean_missing_values(df, strategy='median')
    print(f"Post-cleaning row count: {len(df)}")
    print("Post-cleaning TotalClaims stats:\\n", df['TotalClaims'].describe())

    numerical_cols = [
        'UnderwrittenCoverID', 'PolicyID', 'IsVATRegistered', 'PostalCode',
        'mmcode', 'RegistrationYear', 'Cylinders', 'cubiccapacity',
        'kilowatts', 'NumberOfDoors', 'CustomValueEstimate',
        'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium'
    ]
    df = df.copy()
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Post-outlier removal row count: {len(df)}")
    print("Post-outlier removal TotalClaims stats:\\n", df['TotalClaims'].describe())

    if drop_leaky_columns and 'LossRatio' in df.columns:
        df = df.drop(columns=['LossRatio'])

    return df


def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

def get_features_and_target(df, target_column='TotalClaims'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def load_clean_data_for_modeling(file_path='../data/raw/MachineLearningRating_v3.txt', target_column='TotalClaims'):
    df = load_raw_data(file_path)
    df_cleaned = preprocess_data(df, drop_leaky_columns=True)
    df_cleaned.to_csv('../data/processed/cleaned_data.csv', index=False)
    X, y = get_features_and_target(df_cleaned, target_column=target_column)
    return df_cleaned, X, y