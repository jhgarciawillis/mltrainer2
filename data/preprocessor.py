import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from core.config import config
from core.exceptions import PreprocessingError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataPreprocessor:
    """Handle all data preprocessing operations."""
    
    def __init__(self):
        self.preprocessing_history: List[Dict[str, Any]] = []
        self.preprocessing_pipelines: Dict[str, Pipeline] = {}
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def create_preprocessing_pipeline(
        self,
        data: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
        scaling_method: str = 'standard',
        imputation_method: str = 'mean',
        encoding_method: str = 'onehot',
        pipeline_name: str = 'default_pipeline'
    ) -> Pipeline:
        """Create preprocessing pipeline based on specified methods."""
        try:
            numeric_transformer = self._create_numeric_pipeline(
                scaling_method,
                imputation_method
            )
            
            categorical_transformer = self._create_categorical_pipeline(
                imputation_method,
                encoding_method
            )
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor)
            ])
            
            # Store pipeline
            self.preprocessing_pipelines[pipeline_name] = pipeline
            
            # Store feature names
            self.feature_names[pipeline_name] = numeric_features + categorical_features
            
            # Record preprocessing configuration
            self._record_preprocessing_config(
                pipeline_name,
                {
                    'numeric_features': numeric_features,
                    'categorical_features': categorical_features,
                    'scaling_method': scaling_method,
                    'imputation_method': imputation_method,
                    'encoding_method': encoding_method
                }
            )
            
            return pipeline
            
        except Exception as e:
            raise PreprocessingError(
                f"Error creating preprocessing pipeline: {str(e)}",
                details={
                    'pipeline_name': pipeline_name,
                    'scaling_method': scaling_method,
                    'imputation_method': imputation_method,
                    'encoding_method': encoding_method
                }
            ) from e
    
    def _create_numeric_pipeline(
        self,
        scaling_method: str,
        imputation_method: str
    ) -> Pipeline:
        """Create preprocessing pipeline for numeric features."""
        steps = []
        
        # Add imputer
        imputer = self._get_imputer(imputation_method)
        steps.append(('imputer', imputer))
        
        # Add scaler
        scaler = self._get_scaler(scaling_method)
        steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    def _create_categorical_pipeline(
        self,
        imputation_method: str,
        encoding_method: str
    ) -> Pipeline:
        """Create preprocessing pipeline for categorical features."""
        steps = []
        
        # Add imputer
        imputer = self._get_imputer(imputation_method, categorical=True)
        steps.append(('imputer', imputer))
        
        # Add encoder
        encoder = self._get_encoder(encoding_method)
        steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _get_scaler(self, method: str) -> Any:
        """Get scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        
        if method not in scalers:
            raise PreprocessingError(
                f"Invalid scaling method: {method}",
                details={'available_methods': list(scalers.keys())}
            )
        
        return scalers[method]
    
    def _get_imputer(
        self,
        method: str,
        categorical: bool = False
    ) -> Any:
        """Get imputer based on method."""
        if categorical:
            return SimpleImputer(
                strategy='constant',
                fill_value='missing'
            )
        
        imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        if method not in imputers:
            raise PreprocessingError(
                f"Invalid imputation method: {method}",
                details={'available_methods': list(imputers.keys())}
            )
        
        return imputers[method]
    
    def _get_encoder(self, method: str) -> Any:
        """Get encoder based on method."""
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
        
        encoders = {
            'onehot': OneHotEncoder(sparse=False, handle_unknown='ignore'),
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder()
        }
        
        if method not in encoders:
            raise PreprocessingError(
                f"Invalid encoding method: {method}",
                details={'available_methods': list(encoders.keys())}
            )
        
        return encoders[method]
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def fit_transform(
        self,
        data: pd.DataFrame,
        pipeline_name: str = 'default_pipeline'
    ) -> pd.DataFrame:
        """Fit and transform data using specified pipeline."""
        try:
            pipeline = self.preprocessing_pipelines.get(pipeline_name)
            if pipeline is None:
                raise PreprocessingError(
                    f"Pipeline not found: {pipeline_name}"
                )
            
            # Fit and transform
            transformed_data = pipeline.fit_transform(data)
            
            # Convert to DataFrame with proper column names
            feature_names = self.feature_names[pipeline_name]
            transformed_df = pd.DataFrame(
                transformed_data,
                index=data.index,
                columns=feature_names
            )
            
            # Record preprocessing step
            self._record_preprocessing_step(
                pipeline_name,
                'fit_transform',
                data.shape,
                transformed_df.shape
            )
            
            return transformed_df
            
        except Exception as e:
            raise PreprocessingError(
                f"Error in fit_transform: {str(e)}",
                details={'pipeline_name': pipeline_name}
            ) from e
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def transform(
        self,
        data: pd.DataFrame,
        pipeline_name: str = 'default_pipeline'
    ) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        try:
            pipeline = self.preprocessing_pipelines.get(pipeline_name)
            if pipeline is None:
                raise PreprocessingError(
                    f"Pipeline not found: {pipeline_name}"
                )
            
            # Transform
            transformed_data = pipeline.transform(data)
            
            # Convert to DataFrame with proper column names
            feature_names = self.feature_names[pipeline_name]
            transformed_df = pd.DataFrame(
                transformed_data,
                index=data.index,
                columns=feature_names
            )
            
            # Record preprocessing step
            self._record_preprocessing_step(
                pipeline_name,
                'transform',
                data.shape,
                transformed_df.shape
            )
            
            return transformed_df
            
        except Exception as e:
            raise PreprocessingError(
                f"Error in transform: {str(e)}",
                details={'pipeline_name': pipeline_name}
            ) from e
    
    def _record_preprocessing_config(
        self,
        pipeline_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Record preprocessing configuration."""
        self.preprocessing_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'pipeline_name': pipeline_name,
            'type': 'configuration',
            'config': config
        })
    
    def _record_preprocessing_step(
        self,
        pipeline_name: str,
        operation: str,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int]
    ) -> None:
        """Record preprocessing step."""
        self.preprocessing_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'pipeline_name': pipeline_name,
            'type': 'operation',
            'operation': operation,
            'input_shape': input_shape,
            'output_shape': output_shape
        })
    
    @monitor_performance
    def save_pipeline(
        self,
        pipeline_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Save preprocessing pipeline."""
        if path is None:
            path = config.directories.base_dir / 'preprocessing'
        
        path.mkdir(parents=True, exist_ok=True)
        
        pipeline = self.preprocessing_pipelines.get(pipeline_name)
        if pipeline is None:
            raise PreprocessingError(
                f"Pipeline not found: {pipeline_name}"
            )
        
        # Save pipeline
        pipeline_path = path / f"{pipeline_name}_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        
        # Save feature names
        feature_names_path = path / f"{pipeline_name}_features.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names[pipeline_name], f)
        
        # Save preprocessing history
        history_path = path / f"{pipeline_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.preprocessing_history, f, indent=4)
        
        logger.info(f"Preprocessing pipeline saved to {path}")
    
    @monitor_performance
    def load_pipeline(
        self,
        pipeline_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Load preprocessing pipeline."""
        if path is None:
            path = config.directories.base_dir / 'preprocessing'
        
        # Load pipeline
        pipeline_path = path / f"{pipeline_name}_pipeline.joblib"
        if not pipeline_path.exists():
            raise PreprocessingError(
                f"Pipeline file not found: {pipeline_path}"
            )
        
        self.preprocessing_pipelines[pipeline_name] = joblib.load(pipeline_path)
        
        # Load feature names
        feature_names_path = path / f"{pipeline_name}_features.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names[pipeline_name] = json.load(f)
        
        # Load preprocessing history
        history_path = path / f"{pipeline_name}_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.preprocessing_history = json.load(f)
        
        logger.info(f"Preprocessing pipeline loaded from {path}")

# Create global preprocessor instance
preprocessor = DataPreprocessor()