import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path
from scipy import stats
from datetime import datetime

from core.config import config
from core.exceptions import AnalysisError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization.plotter import plotter

class DataAnalyzer:
    """Handle all data analysis operations."""
    
    def __init__(self):
        self.analysis_results: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.statistical_tests: Dict[str, Dict[str, Any]] = {}
        self.correlations: Dict[str, pd.DataFrame] = {}
        self.summaries: Dict[str, pd.DataFrame] = {}
        
    @monitor_performance
    @handle_exceptions(AnalysisError)
    def analyze_dataset(
        self,
        data: pd.DataFrame,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        try:
            # Get default config if none provided
            if analysis_config is None:
                analysis_config = self._get_default_analysis_config()
            
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            results = {
                'basic_stats': self.calculate_basic_statistics(data),
                'distributions': self.analyze_distributions(data),
                'correlations': self.analyze_correlations(data),
                'outliers': self.detect_outliers(data),
                'missing_values': self.analyze_missing_values(data),
                'unique_values': self.analyze_unique_values(data),
                'data_types': self.analyze_data_types(data),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'rows': len(data),
                    'columns': len(data.columns)
                }
            }
            
            # Store results
            self.analysis_results[analysis_id] = results
            
            # Record analysis
            self._record_analysis(analysis_id, analysis_config)
            
            return results
            
        except Exception as e:
            raise AnalysisError(
                f"Error performing data analysis: {str(e)}"
            ) from e
    
    @monitor_performance
    def calculate_basic_statistics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate basic statistical measures."""
        stats = {}
        
        # Numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            stats['numeric'] = {
                'summary': data[numeric_cols].describe().to_dict(),
                'skewness': data[numeric_cols].skew().to_dict(),
                'kurtosis': data[numeric_cols].kurtosis().to_dict()
            }
            
            # Additional statistics
            for col in numeric_cols:
                stats['numeric'][col] = {
                    'mode': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'range': data[col].max() - data[col].min(),
                    'iqr': data[col].quantile(0.75) - data[col].quantile(0.25),
                    'coefficient_variation': data[col].std() / data[col].mean() if data[col].mean() != 0 else None
                }
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            stats['categorical'] = {}
            for col in categorical_cols:
                value_counts = data[col].value_counts()
                stats['categorical'][col] = {
                    'unique_count': data[col].nunique(),
                    'mode': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'frequencies': value_counts.to_dict(),
                    'proportions': (value_counts / len(data)).to_dict()
                }
        
        return stats
    
    @monitor_performance
    def analyze_distributions(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze data distributions."""
        distributions = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Basic distribution metrics
            dist_metrics = {
                'mean': float(data[col].mean()),
                'median': float(data[col].median()),
                'std': float(data[col].std()),
                'skewness': float(stats.skew(data[col].dropna())),
                'kurtosis': float(stats.kurtosis(data[col].dropna()))
            }
            
            # Normality tests
            normality_test = stats.normaltest(data[col].dropna())
            dist_metrics['normality_test'] = {
                'statistic': float(normality_test.statistic),
                'p_value': float(normality_test.pvalue)
            }
            
            # Histogram data
            hist, bin_edges = np.histogram(data[col].dropna(), bins='auto')
            dist_metrics['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
            
            distributions[col] = dist_metrics
            
            # Create distribution plot
            fig = plotter.create_plot(
                'histogram',
                data=data,
                x=col,
                title=f'Distribution of {col}'
            )
            distributions[f'{col}_plot'] = fig
        
        return distributions
    
    @monitor_performance
    def analyze_correlations(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between features."""
        if methods is None:
            methods = ['pearson', 'spearman']
        
        correlations = {}
        numeric_data = data.select_dtypes(include=[np.number])
        
        for method in methods:
            corr_matrix = numeric_data.corr(method=method)
            correlations[method] = corr_matrix.to_dict()
            
            # Store correlation matrix
            self.correlations[method] = corr_matrix
            
            # Create correlation plot
            fig = plotter.create_plot(
                'heatmap',
                data=corr_matrix,
                title=f'{method.capitalize()} Correlation Matrix'
            )
            correlations[f'{method}_plot'] = fig
        
        return correlations
    
    @monitor_performance
    def detect_outliers(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        if methods is None:
            methods = ['zscore', 'iqr']
        
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_outliers = {}
            
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                col_outliers['zscore'] = {
                    'indices': np.where(z_scores > 3)[0].tolist(),
                    'values': data[col][z_scores > 3].tolist()
                }
            
            if 'iqr' in methods:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
                col_outliers['iqr'] = {
                    'indices': data[outlier_mask].index.tolist(),
                    'values': data[col][outlier_mask].tolist()
                }
            
            # Create box plot
            fig = plotter.create_plot(
                'box',
                data=data,
                y=col,
                title=f'Box Plot of {col}'
            )
            col_outliers['plot'] = fig
            
            outliers[col] = col_outliers
        
        return outliers
    
    @monitor_performance
    def analyze_missing_values(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        missing = {
            'total_missing': int(data.isnull().sum().sum()),
            'missing_percentage': float((data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100),
            'columns': {}
        }
        
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            missing['columns'][col] = {
                'count': int(col_missing),
                'percentage': float((col_missing / len(data)) * 100)
            }
        
        # Create missing values plot
        missing_data = pd.DataFrame({
            'Column': list(missing['columns'].keys()),
            'Missing Count': [info['count'] for info in missing['columns'].values()]
        })
        fig = plotter.create_plot(
            'bar',
            data=missing_data,
            x='Column',
            y='Missing Count',
            title='Missing Values by Column'
        )
        missing['plot'] = fig
        
        return missing
    
    @monitor_performance
    def analyze_unique_values(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze unique values in each column."""
        unique_values = {}
        
        for col in data.columns:
            unique_counts = data[col].value_counts()
            unique_values[col] = {
                'count': int(data[col].nunique()),
                'percentage': float((data[col].nunique() / len(data)) * 100),
                'top_values': unique_counts.head().to_dict()
            }
            
            if pd.api.types.is_numeric_dtype(data[col]):
                unique_values[col]['min'] = float(data[col].min())
                unique_values[col]['max'] = float(data[col].max())
        
        return unique_values
    
    @monitor_performance
    def analyze_data_types(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze data types and their properties."""
        type_analysis = {
            'type_counts': data.dtypes.value_counts().to_dict(),
            'columns': {}
        }
        
        for col in data.columns:
            type_analysis['columns'][col] = {
                'dtype': str(data[col].dtype),
                'memory_usage': int(data[col].memory_usage(deep=True)),
                'is_numeric': pd.api.types.is_numeric_dtype(data[col]),
                'is_categorical': pd.api.types.is_categorical_dtype(data[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(data[col])
            }
        
        return type_analysis
    
    def _get_default_analysis_config(self) -> Dict[str, Any]:
        """Get default analysis configuration."""
        return {
            'basic_stats': True,
            'distributions': True,
            'correlations': {
                'enabled': True,
                'methods': ['pearson', 'spearman']
            },
            'outliers': {
                'enabled': True,
                'methods': ['zscore', 'iqr']
            },
            'missing_values': True,
            'unique_values': True,
            'data_types': True
        }
    
    def _record_analysis(
        self,
        analysis_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Record analysis in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'analysis_id': analysis_id,
            'configuration': config
        }
        
        self.analysis_history.append(record)
        state_manager.set_state(
            f'analysis.history.{len(self.analysis_history)}',
            record
        )

# Create global analyzer instance
analyzer = DataAnalyzer()