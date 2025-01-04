from typing import Optional, Any, Dict
from datetime import datetime

class MLTrainerException(Exception):
    """Base exception class for ML Trainer application."""
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        previous_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or 'UNKNOWN_ERROR'
        self.details = details or {}
        self.previous_exception = previous_exception
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            'error_code': self.code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'previous_exception': str(self.previous_exception) if self.previous_exception else None
        }

class ConfigurationError(MLTrainerException):
    """Raised when there's an error in configuration."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'CONFIG_ERROR', details)

class DataError(MLTrainerException):
    """Base class for data-related errors."""
    def __init__(self, message: str, code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, details)

class DataLoadError(DataError):
    """Raised when there's an error loading data."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'DATA_LOAD_ERROR', details)

class DataValidationError(DataError):
    """Raised when data validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'DATA_VALIDATION_ERROR', details)

class PreprocessingError(DataError):
    """Raised when there's an error in preprocessing."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'PREPROCESSING_ERROR', details)

class FeatureEngineeringError(MLTrainerException):
    """Raised when there's an error in feature engineering."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'FEATURE_ENGINEERING_ERROR', details)

class ClusteringError(MLTrainerException):
    """Raised when there's an error in clustering."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'CLUSTERING_ERROR', details)

class ModelError(MLTrainerException):
    """Base class for model-related errors."""
    def __init__(self, message: str, code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, details)

class ModelTrainingError(ModelError):
    """Raised when there's an error during model training."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'MODEL_TRAINING_ERROR', details)

class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'MODEL_VALIDATION_ERROR', details)

class PredictionError(MLTrainerException):
    """Raised when there's an error during prediction."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'PREDICTION_ERROR', details)

class VisualizationError(MLTrainerException):
    """Raised when there's an error in visualization."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'VISUALIZATION_ERROR', details)

class StateError(MLTrainerException):
    """Raised when there's an error in state management."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'STATE_ERROR', details)

class UIError(MLTrainerException):
    """Raised when there's an error in UI operations."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'UI_ERROR', details)

class ValidationError(MLTrainerException):
    """Raised when validation fails in any component."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'VALIDATION_ERROR', details)

class ExportError(MLTrainerException):
    """Raised when there's an error exporting data or results."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'EXPORT_ERROR', details)

class ImportError(MLTrainerException):
    """Raised when there's an error importing data or models."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'IMPORT_ERROR', details)

def handle_exception(exception: Exception) -> Dict[str, Any]:
    """Convert any exception to a standardized format."""
    if isinstance(exception, MLTrainerException):
        return exception.to_dict()
    return MLTrainerException(
        message=str(exception),
        code='UNKNOWN_ERROR',
        previous_exception=exception
    ).to_dict()