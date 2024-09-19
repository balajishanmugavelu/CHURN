import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def separate_features_target(df: pd.DataFrame, target_column: str):
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    else:
        return df, None
