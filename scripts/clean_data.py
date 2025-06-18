import pandas as pd


def clean_category(category):
    """
    Standardizes the CoverCategory column.
    """
    if pd.isna(category):
        return 'Unknown'
    
    category = str(category).strip().lower()
    category = category.replace('.', '').replace('-', '').replace(' (2015)', '')

    category_map = {
        'owndamage': 'Own Damage',
        'windscreen': 'Windscreen',
        'incomprotector': 'Income Protector',
        'creditprotection': 'Credit Protection'
    }

    return category_map.get(category, category.title())


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary columns from the DataFrame.
    """
    cols_to_drop = [col for col in ["NumberOfVehiclesInFleet", "CrossBorder"] if col in df.columns]
    return df.drop(columns=cols_to_drop)


def clean_missing_values(df: pd.DataFrame, strategy='mean') -> pd.DataFrame:
    df_cleaned = df.copy()

    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    else:
        numeric_cols = df_cleaned.select_dtypes(include='number').columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().any() and col != 'TotalClaims':  # Exclude TotalClaims
                if strategy == 'mean':
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif strategy == 'median':
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        # Handle TotalClaims separately (e.g., keep NaN or set to 0 only if intended)
        if 'TotalClaims' in df_cleaned.columns and df_cleaned['TotalClaims'].isnull().any():
            df_cleaned['TotalClaims'] = df_cleaned['TotalClaims'].fillna(0)  # Assume missing = no claim

    return df_cleaned

def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Removes outliers using IQR method from specified numeric columns.
    Ignores non-numeric or boolean-type columns safely.
    """
    df_cleaned = df.copy()

    for col in cols:
        if col in df_cleaned.columns:
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue  # skip non-numeric columns

            # Convert boolean to integer if needed
            if pd.api.types.is_bool_dtype(df_cleaned[col]):
                df_cleaned[col] = df_cleaned[col].astype(int)

            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
            ]

    return df_cleaned
