import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from pathlib import Path
from datetime import datetime
import json

from core.config import config
from core.exceptions import DataValidationError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataValidator:
    """Handle all data validation operations."""
    
    def __init__(self):
        self.validation_reports: Dict[str, Dict[str, Any]] = {}
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []
    
    @monitor_performance
    @handle_exceptions(DataValidationError)
    def validate_dataset(
        self,
        data: pd.DataFrame,
        rules: Optional[Dict[str, Any]] = None,
        dataset_name: str = "unnamed_dataset"
    ) -> Dict[str, Any]:
        """Validate entire dataset against specified rules."""
        if rules is None:
            rules = self._get_default_rules()
        
        validation_results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'checks': {}
        }
        
        try:
            # Basic dataset validation
            self._validate_basic_requirements(data, validation_results)
            
            # Column validation
            self._validate_columns(data, rules, validation_results)
            
            # Data type validation
            self._validate_datatypes(data, rules, validation_results)
            
            # Value range validation
            self._validate_value_ranges(data, rules, validation_results)
            
            # Missing values validation
            self._validate_missing_values(data, rules, validation_results)
            
            # Duplicate validation
            self._validate_duplicates(data, rules, validation_results)
            
            # Custom validation rules
            if 'custom_rules' in rules:
                self._apply_custom_rules(data, rules['custom_rules'], validation_results)
            
            # Calculate statistics
            validation_results['statistics'] = self._calculate_statistics(data)
            
            # Store validation results
            self.validation_reports[dataset_name] = validation_results
            self.validation_history.append(validation_results)
            
            # Update state
            state_manager.set_state(
                f'data.validation.{dataset_name}',
                validation_results
            )
            
            return validation_results
            
        except Exception as e:
            raise DataValidationError(
                f"Validation failed: {str(e)}",
                details={'dataset_name': dataset_name}
            ) from e
    
    def _validate_basic_requirements(
        self,
        data: pd.DataFrame,
        results: Dict[str, Any]
    ) -> None:
        """Validate basic dataset requirements."""
        if data.empty:
            results['passed'] = False
            results['errors'].append("Dataset is empty")
            return
        
        if data.columns.empty:
            results['passed'] = False
            results['errors'].append("Dataset has no columns")
            return
        
        results['checks']['basic_requirements'] = {
            'rows': len(data),
            'columns': len(data.columns),
            'passed': True
        }
    
    def _validate_columns(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Validate column requirements."""
        column_checks = {}
        
        # Required columns
        if 'required_columns' in rules:
            missing_cols = set(rules['required_columns']) - set(data.columns)
            if missing_cols:
                results['passed'] = False
                results['errors'].append(f"Missing required columns: {missing_cols}")
                column_checks['required_columns'] = {
                    'passed': False,
                    'missing': list(missing_cols)
                }
        
        # Column naming
        if 'column_pattern' in rules:
            pattern = rules['column_pattern']
            invalid_cols = [
                col for col in data.columns 
                if not pd.Series([col]).str.match(pattern).bool()
            ]
            if invalid_cols:
                results['warnings'].append(
                    f"Columns not matching pattern {pattern}: {invalid_cols}"
                )
                column_checks['column_pattern'] = {
                    'passed': False,
                    'invalid_columns': invalid_cols
                }
        
        results['checks']['columns'] = column_checks
    
    def _validate_datatypes(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Validate data types."""
        dtype_checks = {}
        
        if 'column_types' in rules:
            for col, expected_type in rules['column_types'].items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    matched = (
                        pd.api.types.is_dtype_equal(actual_type, expected_type) or
                        (str(actual_type) == expected_type)
                    )
                    if not matched:
                        results['warnings'].append(
                            f"Column {col} has type {actual_type}, expected {expected_type}"
                        )
                        dtype_checks[col] = {
                            'passed': False,
                            'expected': str(expected_type),
                            'actual': str(actual_type)
                        }
        
        results['checks']['datatypes'] = dtype_checks
    
    def _validate_value_ranges(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Validate value ranges for numeric columns."""
        range_checks = {}
        
        if 'value_ranges' in rules:
            for col, ranges in rules['value_ranges'].items():
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    min_val = ranges.get('min')
                    max_val = ranges.get('max')
                    
                    if min_val is not None and data[col].min() < min_val:
                        results['errors'].append(
                            f"Column {col} contains values below minimum {min_val}"
                        )
                        range_checks[col] = {
                            'passed': False,
                            'min_violation': True,
                            'min_value': data[col].min()
                        }
                    
                    if max_val is not None and data[col].max() > max_val:
                        results['errors'].append(
                            f"Column {col} contains values above maximum {max_val}"
                        )
                        range_checks[col] = {
                            'passed': False,
                            'max_violation': True,
                            'max_value': data[col].max()
                        }
        
        results['checks']['value_ranges'] = range_checks
    
    def _validate_missing_values(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Validate missing values."""
        missing_checks = {}
        
        if 'missing_values' in rules:
            max_missing_pct = rules['missing_values'].get('max_percent', 100)
            
            for col in data.columns:
                missing_pct = (data[col].isnull().sum() / len(data)) * 100
                if missing_pct > max_missing_pct:
                    results['warnings'].append(
                        f"Column {col} has {missing_pct:.2f}% missing values"
                    )
                    missing_checks[col] = {
                        'passed': False,
                        'missing_percent': missing_pct
                    }
        
        results['checks']['missing_values'] = missing_checks
    
    def _validate_duplicates(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Validate duplicate values."""
        duplicate_checks = {}
        
        if 'duplicates' in rules:
            # Full row duplicates
            if rules['duplicates'].get('check_full_rows', True):
                duplicates = data.duplicated()
                if duplicates.any():
                    results['warnings'].append(
                        f"Found {duplicates.sum()} duplicate rows"
                    )
                    duplicate_checks['full_rows'] = {
                        'passed': False,
                        'count': int(duplicates.sum())
                    }
            
            # Column-specific duplicates
            if 'unique_columns' in rules['duplicates']:
                for col in rules['duplicates']['unique_columns']:
                    if col in data.columns:
                        duplicates = data[col].duplicated()
                        if duplicates.any():
                            results['warnings'].append(
                                f"Column {col} has {duplicates.sum()} duplicate values"
                            )
                            duplicate_checks[col] = {
                                'passed': False,
                                'count': int(duplicates.sum())
                            }
        
        results['checks']['duplicates'] = duplicate_checks
    
    def _apply_custom_rules(
        self,
        data: pd.DataFrame,
        custom_rules: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> None:
        """Apply custom validation rules."""
        custom_checks = {}
        
        for rule in custom_rules:
            rule_name = rule.get('name', 'unnamed_rule')
            condition = rule.get('condition')
            
            if condition:
                try:
                    # Safely evaluate condition
                    mask = eval(condition, {'df': data, 'np': np, 'pd': pd})
                    if isinstance(mask, pd.Series) and not mask.all():
                        results['warnings'].append(
                            f"Custom rule '{rule_name}' failed"
                        )
                        custom_checks[rule_name] = {
                            'passed': False,
                            'failure_count': (~mask).sum()
                        }
                except Exception as e:
                    logger.error(f"Error evaluating custom rule {rule_name}: {str(e)}")
        
        results['checks']['custom_rules'] = custom_checks
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'column_types': data.dtypes.value_counts().to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['numeric_stats'][col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'quartiles': [
                    float(q) for q in data[col].quantile([0.25, 0.5, 0.75])
                ]
            }
        
        return stats
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default validation rules."""
        return {
            'missing_values': {
                'max_percent': 50
            },
            'duplicates': {
                'check_full_rows': True
            },
            'column_types': {},
            'value_ranges': {},
            'custom_rules': []
        }
    
    @monitor_performance
    def get_validation_summary(
        self,
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get validation summary for specific dataset or all datasets."""
        if dataset_name:
            return self.validation_reports.get(dataset_name, {})
        
        return {
            'total_validations': len(self.validation_history),
            'passed_validations': sum(
                1 for report in self.validation_history 
                if report['passed']
            ),
            'latest_validations': self.validation_history[-5:],
            'validation_reports': self.validation_reports
        }
    
    @monitor_performance
    def save_validation_results(
        self,
        path: Optional[Path] = None
    ) -> None:
        """Save validation results to disk."""
        if path is None:
            path = config.directories.base_dir / 'validation_results'
        
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save latest results
        results_file = path / f'validation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.validation_reports, f, indent=4)
        
        # Save history
        history_file = path / f'validation_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            json.dump(self.validation_history, f, indent=4)
        
        logger.info(f"Validation results saved to {path}")

# Create global data validator instance
data_validator = DataValidator()