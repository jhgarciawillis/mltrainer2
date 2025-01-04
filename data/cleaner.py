import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path
from datetime import datetime
import re

from core.config import config
from core.exceptions import DataCleaningError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataCleaner:
    """Handle all data cleaning operations."""
    
    def __init__(self):
        self.cleaning_history: List[Dict[str, Any]] = []
        self.cleaning_stats: Dict[str, Dict[str, Any]] = {}
        self.original_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
    
    @monitor_performance
    @handle_exceptions(DataCleaningError)
    def clean_dataset(
        self,
        data: pd.DataFrame,
        cleaning_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Clean dataset according to specified configuration."""
        try:
            # Store original data
            self.original_data = data.copy()
            cleaned_data = data.copy()
            
            # Get default config if none provided
            if cleaning_config is None:
                cleaning_config = self._get_default_cleaning_config()
            
            # Apply cleaning operations in order
            cleaning_steps = [
                ('remove_duplicates', self._remove_duplicates),
                ('handle_missing_values', self._handle_missing_values),
                ('fix_datatypes', self._fix_datatypes),
                ('handle_outliers', self._handle_outliers),
                ('standardize_text', self._standardize_text),
                ('validate_values', self._validate_values),
                ('fix_inconsistencies', self._fix_inconsistencies)
            ]
            
            for step_name, step_func in cleaning_steps:
                if cleaning_config.get(step_name, {}).get('enabled', True):
                    cleaned_data = step_func(
                        cleaned_data,
                        cleaning_config.get(step_name, {})
                    )
                    self._record_cleaning_step(
                        step_name,
                        len(self.original_data),
                        len(cleaned_data)
                    )
            
            # Store cleaned data
            self.cleaned_data = cleaned_data
            
            # Calculate and store cleaning statistics
            self._calculate_cleaning_stats()
            
            return cleaned_data
            
        except Exception as e:
            raise DataCleaningError(
                f"Error during data cleaning: {str(e)}"
            ) from e
    
    @monitor_performance
    def _remove_duplicates(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Remove duplicate rows based on configuration."""
        subset = config.get('subset', None)
        keep = config.get('keep', 'first')
        
        initial_rows = len(data)
        cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(cleaned_data)
        
        logger.info(f"Removed {removed_rows} duplicate rows")
        return cleaned_data
    
    @monitor_performance
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        strategy = config.get('strategy', 'remove')
        threshold = config.get('threshold', 0.5)
        fill_values = config.get('fill_values', {})
        
        if strategy == 'remove':
            # Remove columns with too many missing values
            missing_ratio = data.isnull().sum() / len(data)
            cols_to_drop = missing_ratio[missing_ratio > threshold].index
            data = data.drop(columns=cols_to_drop)
            
            # Remove rows with any remaining missing values
            data = data.dropna()
            
        elif strategy == 'fill':
            for col in data.columns:
                if col in fill_values:
                    data[col].fillna(fill_values[col], inplace=True)
                elif pd.api.types.is_numeric_dtype(data[col]):
                    data[col].fillna(data[col].mean(), inplace=True)
                else:
                    data[col].fillna(data[col].mode().iloc[0], inplace=True)
        
        return data
    
    @monitor_performance
    def _fix_datatypes(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fix data types based on configuration."""
        type_mappings = config.get('type_mappings', {})
        infer_types = config.get('infer_types', True)
        
        # Apply explicit type mappings
        for col, dtype in type_mappings.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        # Infer and fix types if enabled
        if infer_types:
            for col in data.columns:
                if col not in type_mappings:
                    data[col] = self._infer_and_convert_type(data[col])
        
        return data
    
    @monitor_performance
    def _handle_outliers(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Handle outliers based on configuration."""
        method = config.get('method', 'zscore')
        threshold = config.get('threshold', 3)
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns)
        
        for col in columns:
            if col in data.columns:
                if method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    data = data[z_scores <= threshold]
                elif method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    data = data[
                        (data[col] >= Q1 - threshold * IQR) &
                        (data[col] <= Q3 + threshold * IQR)
                    ]
        
        return data
    
    @monitor_performance
    def _standardize_text(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Standardize text data based on configuration."""
        columns = config.get('columns', data.select_dtypes(include=['object']).columns)
        operations = config.get('operations', ['lowercase', 'strip', 'spaces'])
        
        for col in columns:
            if col in data.columns and pd.api.types.is_string_dtype(data[col]):
                if 'lowercase' in operations:
                    data[col] = data[col].str.lower()
                if 'strip' in operations:
                    data[col] = data[col].str.strip()
                if 'spaces' in operations:
                    data[col] = data[col].str.replace(r'\s+', ' ', regex=True)
        
        return data
    
    @monitor_performance
    def _validate_values(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Validate values based on configuration."""
        validations = config.get('validations', {})
        
        for col, rules in validations.items():
            if col in data.columns:
                if 'range' in rules:
                    min_val, max_val = rules['range']
                    data = data[
                        (data[col] >= min_val) &
                        (data[col] <= max_val)
                    ]
                if 'pattern' in rules:
                    pattern = rules['pattern']
                    data = data[data[col].str.match(pattern)]
                if 'allowed_values' in rules:
                    allowed = set(rules['allowed_values'])
                    data = data[data[col].isin(allowed)]
        
        return data
    
    @monitor_performance
    def _fix_inconsistencies(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fix inconsistencies based on configuration."""
        replacements = config.get('replacements', {})
        
        for col, mapping in replacements.items():
            if col in data.columns:
                data[col] = data[col].replace(mapping)
        
        return data
    
    def _infer_and_convert_type(self, series: pd.Series) -> pd.Series:
        """Infer and convert data type for a series."""
        # Try numeric conversion
        try:
            return pd.to_numeric(series)
        except:
            pass
        
        # Try datetime conversion
        try:
            return pd.to_datetime(series)
        except:
            pass
        
        # Try boolean conversion
        if series.nunique() <= 2:
            try:
                return series.astype(bool)
            except:
                pass
        
        # Keep as string if no other type fits
        return series.astype(str)
    
    def _get_default_cleaning_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'remove_duplicates': {
                'enabled': True,
                'subset': None,
                'keep': 'first'
            },
            'handle_missing_values': {
                'enabled': True,
                'strategy': 'remove',
                'threshold': 0.5,
                'fill_values': {}
            },
            'fix_datatypes': {
                'enabled': True,
                'infer_types': True,
                'type_mappings': {}
            },
            'handle_outliers': {
                'enabled': True,
                'method': 'zscore',
                'threshold': 3,
                'columns': []
            },
            'standardize_text': {
                'enabled': True,
                'operations': ['lowercase', 'strip', 'spaces'],
                'columns': []
            },
            'validate_values': {
                'enabled': True,
                'validations': {}
            },
            'fix_inconsistencies': {
                'enabled': True,
                'replacements': {}
            }
        }
    
    def _record_cleaning_step(
        self,
        step_name: str,
        initial_rows: int,
        final_rows: int
    ) -> None:
        """Record cleaning step details."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': initial_rows - final_rows
        }
        
        self.cleaning_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'data.cleaning.history.{len(self.cleaning_history)}',
            record
        )
    
    def _calculate_cleaning_stats(self) -> None:
        """Calculate cleaning statistics."""
        if self.original_data is None or self.cleaned_data is None:
            return
        
        self.cleaning_stats = {
            'initial_rows': len(self.original_data),
            'final_rows': len(self.cleaned_data),
            'rows_removed': len(self.original_data) - len(self.cleaned_data),
            'initial_columns': len(self.original_data.columns),
            'final_columns': len(self.cleaned_data.columns),
            'columns_removed': len(self.original_data.columns) - len(self.cleaned_data.columns),
            'null_values_removed': (
                self.original_data.isnull().sum().sum() -
                self.cleaned_data.isnull().sum().sum()
            ),
            'cleaning_steps': len(self.cleaning_history),
            'cleaning_duration': sum(
                record.get('duration', 0)
                for record in self.cleaning_history
            )
        }
        
        # Update state
        state_manager.set_state('data.cleaning.stats', self.cleaning_stats)

# Create global data cleaner instance
data_cleaner = DataCleaner()