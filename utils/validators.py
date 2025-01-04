import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from pathlib import Path

from core.exceptions import ValidationError
from core.config import config
from utils.decorators import log_execution, handle_exceptions

@handle_exceptions(ValidationError)
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_rows: Optional[int] = None,
    allow_nulls: bool = False
) -> bool:
    """Validate DataFrame against specified requirements."""
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValidationError("DataFrame is empty")
    
    # Validate row count
    row_count = len(df)
    if row_count < min_rows:
        raise ValidationError(
            f"DataFrame has fewer rows ({row_count}) than required ({min_rows})"
        )
    if max_rows and row_count > max_rows:
        raise ValidationError(
            f"DataFrame has more rows ({row_count}) than allowed ({max_rows})"
        )
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}"
            )
    
    # Validate numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col not in df.columns:
                raise ValidationError(f"Numeric column {col} not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column {col} is not numeric")
    
    # Validate categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col not in df.columns:
                raise ValidationError(f"Categorical column {col} not found in DataFrame")
    
    # Check for nulls if not allowed
    if not allow_nulls and df.isnull().any().any():
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        raise ValidationError(
            "Null values found in DataFrame",
            {'columns_with_nulls': columns_with_nulls.to_dict()}
        )
    
    return True

@handle_exceptions(ValidationError)
def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: Optional[Set[str]] = None,
    must_exist: bool = True,
    create_parent: bool = False,
    max_size: Optional[int] = None
) -> bool:
    """Validate file path and properties."""
    
    # Convert to Path object
    path = Path(file_path)
    
    # Check parent directory
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif must_exist and not path.parent.exists():
        raise ValidationError(f"Parent directory does not exist: {path.parent}")
    
    # Check file existence
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")
    
    # Validate extension
    if allowed_extensions:
        if path.suffix.lstrip('.').lower() not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension. Allowed: {allowed_extensions}"
            )
    
    # Check file size if it exists and max_size is specified
    if max_size and path.exists():
        size = path.stat().st_size
        if size > max_size:
            raise ValidationError(
                f"File size ({size} bytes) exceeds maximum ({max_size} bytes)"
            )
    
    return True

@handle_exceptions(ValidationError)
def validate_column_names(
    columns: List[str],
    unique: bool = True,
    max_length: Optional[int] = None,
    allowed_characters: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
) -> bool:
    """Validate column names against specified criteria."""
    
    if not columns:
        raise ValidationError("Empty column list")
    
    # Check uniqueness
    if unique and len(columns) != len(set(columns)):
        duplicates = [col for col in columns if columns.count(col) > 1]
        raise ValidationError(f"Duplicate column names found: {duplicates}")
    
    for col in columns:
        # Check length
        if max_length and len(col) > max_length:
            raise ValidationError(
                f"Column name '{col}' exceeds maximum length of {max_length}"
            )
        
        # Check allowed characters
        if allowed_characters and not all(c in allowed_characters for c in col):
            raise ValidationError(
                f"Column name '{col}' contains invalid characters"
            )
        
        # Check prefix
        if prefix and not col.startswith(prefix):
            raise ValidationError(
                f"Column name '{col}' does not start with required prefix '{prefix}'"
            )
        
        # Check suffix
        if suffix and not col.startswith(suffix):
            raise ValidationError(
                f"Column name '{col}' does not end with required suffix '{suffix}'"
            )
    
    return True

@handle_exceptions(ValidationError)
def validate_numeric_range(
    values: Union[pd.Series, np.ndarray, List[float]],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_inf: bool = False,
    allow_nan: bool = False
) -> bool:
    """Validate numeric values against range and other criteria."""
    
    # Convert to numpy array for consistent handling
    arr = np.array(values)
    
    if not allow_nan and np.isnan(arr).any():
        raise ValidationError("NaN values found")
    
    if not allow_inf and np.isinf(arr).any():
        raise ValidationError("Infinite values found")
    
    non_null = arr[~np.isnan(arr)]
    
    if min_value is not None and (non_null < min_value).any():
        raise ValidationError(f"Values below minimum {min_value} found")
    
    if max_value is not None and (non_null > max_value).any():
        raise ValidationError(f"Values above maximum {max_value} found")
    
    return True

@handle_exceptions(ValidationError)
def validate_categorical_values(
    values: Union[pd.Series, List[str]],
    allowed_values: Optional[Set[str]] = None,
    min_categories: Optional[int] = None,
    max_categories: Optional[int] = None,
    allow_nulls: bool = False
) -> bool:
    """Validate categorical values against specified criteria."""
    
    # Convert to pandas Series for consistent handling
    series = pd.Series(values)
    
    # Check nulls
    if not allow_nulls and series.isnull().any():
        raise ValidationError("Null values found")
    
    # Get unique non-null values
    unique_values = set(series.dropna().unique())
    
    # Check allowed values
    if allowed_values:
        invalid_values = unique_values - allowed_values
        if invalid_values:
            raise ValidationError(f"Invalid values found: {invalid_values}")
    
    # Check category count
    category_count = len(unique_values)
    if min_categories and category_count < min_categories:
        raise ValidationError(
            f"Too few categories ({category_count} < {min_categories})"
        )
    if max_categories and category_count > max_categories:
        raise ValidationError(
            f"Too many categories ({category_count} > {max_categories})"
        )
    
    return True

@handle_exceptions(ValidationError)
def validate_model_config(
    model_params: Dict[str, Any],
    required_params: Set[str],
    param_types: Dict[str, type],
    param_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> bool:
    """Validate model configuration parameters."""
    
    # Check required parameters
    missing_params = required_params - set(model_params.keys())
    if missing_params:
        raise ValidationError(f"Missing required parameters: {missing_params}")
    
    # Validate parameter types
    for param, expected_type in param_types.items():
        if param in model_params:
            value = model_params[param]
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"Invalid type for {param}: expected {expected_type}, got {type(value)}"
                )
    
    # Validate parameter ranges
    if param_ranges:
        for param, (min_val, max_val) in param_ranges.items():
            if param in model_params:
                value = model_params[param]
                if value < min_val or value > max_val:
                    raise ValidationError(
                        f"Parameter {param} value {value} outside range [{min_val}, {max_val}]"
                    )
    
    return True

@log_execution()
def validate_clustering_input(
    data: pd.DataFrame,
    clustering_config: Dict[str, Dict[str, Any]]
) -> bool:
    """Validate clustering input data and configuration."""
    try:
        # Validate basic DataFrame
        validate_dataframe(data)
        
        # Validate columns used in clustering
        columns_to_cluster = set()
        for col_config in clustering_config.values():
            if 'columns' in col_config:
                columns_to_cluster.update(col_config['columns'])
        
        missing_columns = columns_to_cluster - set(data.columns)
        if missing_columns:
            raise ValidationError(f"Missing columns for clustering: {missing_columns}")
        
        # Validate numeric requirements
        for col in columns_to_cluster:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValidationError(f"Column {col} must be numeric for clustering")
        
        # Validate clustering parameters
        for col, params in clustering_config.items():
            if 'method' in params:
                method = params['method']
                if method not in config.clustering.available_methods:
                    raise ValidationError(f"Invalid clustering method: {method}")
        
        return True
        
    except Exception as e:
        raise ValidationError("Clustering validation failed", {'error': str(e)}) from e

@log_execution()
def validate_training_input(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Dict[str, Any]
) -> bool:
    """Validate training input data and configuration."""
    try:
        # Validate feature DataFrame
        validate_dataframe(X)
        
        # Validate target series
        if y.empty:
            raise ValidationError("Empty target variable")
        
        if len(X) != len(y):
            raise ValidationError(
                f"Feature and target length mismatch: {len(X)} != {len(y)}"
            )
        
        # Validate model configuration
        required_model_params = {'model_type', 'hyperparameters'}
        validate_model_config(
            model_config,
            required_model_params,
            {'model_type': str, 'hyperparameters': dict}
        )
        
        return True
        
    except Exception as e:
        raise ValidationError("Training validation failed", {'error': str(e)}) from e