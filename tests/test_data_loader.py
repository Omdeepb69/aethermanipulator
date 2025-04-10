```python
import pytest
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.testing import assert_frame_equal, assert_series_equal

# --- Start: Hypothetical data_loader module content ---
# In a real scenario, these functions would be in a separate data_loader.py file
# and imported here. For this self-contained example, they are defined directly.

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"File not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            # pandas.read_csv might return an empty DataFrame for files with only headers
            # or handle EmptyDataError for truly empty files.
            # We explicitly check if the resulting DataFrame is empty after loading.
            raise ValueError("CSV file is empty or contains only headers.")
        return df
    except pd.errors.EmptyDataError:
         raise ValueError("CSV file is empty.")
    except Exception as e:
        # Catching potential parsing errors or other read_csv issues
        raise ValueError(f"Error loading CSV: {e}")

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: list = None) -> pd.DataFrame:
    """Handles missing values using SimpleImputer."""
    df_copy = df.copy()
    if columns is None:
        columns_to_impute = df_copy.select_dtypes(include=np.number).columns.tolist()
        if not columns_to_impute: # Handle case where no numeric columns exist
             return df_copy # No imputation needed/possible with default settings
    else:
        columns_to_impute = [col for col in columns if col in df_copy.columns]
        if not columns_to_impute:
            return df_copy # No specified columns found in DataFrame

    numeric_cols_to_impute = df_copy[columns_to_impute].select_dtypes(include=np.number).columns
    categorical_cols_to_impute = df_copy[columns_to_impute].select_dtypes(exclude=np.number).columns

    if not numeric_cols_to_impute.empty:
        numeric_strategy = strategy if strategy in ['mean', 'median', 'constant'] else 'mean'
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df_copy[numeric_cols_to_impute] = numeric_imputer.fit_transform(df_copy[numeric_cols_to_impute])

    if not categorical_cols_to_impute.empty:
        # 'mean' and 'median' are not valid for categorical, default to 'most_frequent'
        categorical_strategy = strategy if strategy in ['most_frequent', 'constant'] else 'most_frequent'
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df_copy[categorical_cols_to_impute] = categorical_imputer.fit_transform(df_copy[categorical_cols_to_impute])

    return df_copy


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows."""
    return df.drop_duplicates()

def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough') # Keep other columns
    return preprocessor

def transform_data(df: pd.DataFrame, preprocessor: ColumnTransformer, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    """Applies the preprocessing pipeline to the data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("Input 'preprocessor' must be a scikit-learn ColumnTransformer.")

    processed_data = preprocessor.fit_transform(df)

    # Get feature names after transformation
    cat_feature_names = []
    try:
        cat_transformer = preprocessor.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        cat_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    except KeyError:
        # 'cat' transformer might not exist if categorical_features is empty
        pass
    except Exception as e:
        # Handle potential errors in getting feature names
        print(f"Warning: Could not retrieve categorical feature names: {e}")
        # Fallback: generate generic names if needed, though ideally get_feature_names_out works
        # num_cat_output_features = processed_data.shape[1] - len(numeric_features) - (len(df.columns) - len(numeric_features) - len(categorical_features) if preprocessor.remainder == 'passthrough' else 0)
        # cat_feature_names = [f"cat_{i}" for i in range(num_cat_output_features)]


    remainder_cols = []
    if preprocessor.remainder == 'passthrough':
         original_cols = list(df.columns)
         processed_cols_set = set(numeric_features + categorical_features)
         remainder_indices = [i for i, col in enumerate(original_cols) if col not in processed_cols_set]
         # Get remainder column names based on original DataFrame columns
         remainder_cols = [original_cols[i] for i in remainder_indices]


    all_feature_names = list(numeric_features) + cat_feature_names + remainder_cols

    # Ensure the number of names matches the number of columns in processed_data
    if len(all_feature_names) != processed_data.shape[1]:
         raise ValueError(f"Mismatch between generated feature names ({len(all_feature_names)}) and processed data columns ({processed_data.shape[1]})")


    processed_df = pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)
