import pandas as pd


def clean_category(category):
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

def clean_df(df: pd):
    df = df.drop(columns=["NumberOfVehiclesInFleet", "CrossBorder"], axis=1)
    return df