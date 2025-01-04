import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Base directory configuration
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DirectoryConfig:
    """Directory configuration with automatic path creation."""
    base_dir: Path = BASE_DIR
    
    # Mode-specific directories
    analysis_dir: Path = field(default_factory=lambda: BASE_DIR / "Analysis")
    training_dir: Path = field(default_factory=lambda: BASE_DIR / "Training")
    prediction_dir: Path = field(default_factory=lambda: BASE_DIR / "Prediction")
    
    # Analysis subdirectories
    analysis_outputs: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Outputs")
    analysis_reports: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Reports")
    analysis_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Visualizations")
    analysis_state: Path = field(default_factory=lambda: BASE_DIR / "Analysis/State")
    analysis_config: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Config")
    
    # Training subdirectories
    training_outputs: Path = field(default_factory=lambda: BASE_DIR / "Training/Outputs")
    training_models: Path = field(default_factory=lambda: BASE_DIR / "Training/Models")
    training_reports: Path = field(default_factory=lambda: BASE_DIR / "Training/Reports")
    training_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Training/Visualizations")
    training_state: Path = field(default_factory=lambda: BASE_DIR / "Training/State")
    training_config: Path = field(default_factory=lambda: BASE_DIR / "Training/Config")
    
    # Prediction subdirectories
    prediction_outputs: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Outputs")
    prediction_reports: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Reports")
    prediction_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Visualizations")
    prediction_results: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Results")

    def __post_init__(self):
        """Create all directories after initialization."""
        self._create_directories()
    
    def _create_directories(self):
        """Create all configured directories."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)

@dataclass
class FileConfig:
    """File-related configurations."""
    # Mandatory model files
    scaler_file: str = "scaler.joblib"
    imputer_file: str = "imputer.joblib"
    trained_model_file: str = "trained_model.joblib"
    cluster_file: str = "cluster.joblib"
    
    # Report files
    data_quality_report: str = "data_quality_report.xlsx"
    performance_metrics_report: str = "performance_metrics.xlsx"
    error_analysis_report: str = "error_analysis.xlsx"
    predictions_file: str = "predictions.xlsx"
    
    # Visualization files
    analysis_visualizations_pdf: str = "analysis_visualizations.pdf"
    training_visualizations_pdf: str = "training_visualizations.pdf"
    prediction_visualizations_pdf: str = "prediction_visualizations.pdf"
    
    # Sheet configuration
    sheet_name_max_length: int = 31
    sheet_name_truncate_suffix: str = "_cluster_db"
    
    # File upload configuration
    allowed_extensions: Set[str] = field(default_factory=lambda: {'csv', 'xlsx'})
    max_file_size: int = 200 * 1024 * 1024  # 200 MB

@dataclass
class ProcessingConfig:
    """Data processing and analysis parameters."""
    # Analysis parameters
    analysis_threshold: float = 0.05
    analysis_metrics: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis'
    ])
    correlation_methods: List[str] = field(default_factory=lambda: [
        'pearson', 'spearman', 'kendall'
    ])
    
    # Outlier parameters
    outlier_threshold: float = 3.0
    
    # Feature engineering parameters
    statistical_agg_functions: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'std'
    ])
    top_k_features: int = 20
    max_interaction_degree: int = 2
    polynomial_degree: int = 2
    feature_selection_score_func: str = 'f_regression'
    
    # Random state
    random_state: int = 42

@dataclass
class ClusteringConfig:
    """Clustering-related configurations."""
    available_methods: List[str] = field(default_factory=lambda: [
        'None', 'DBSCAN', 'KMeans'
    ])
    default_method: str = 'None'
    
    dbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'eps': 0.5,
        'min_samples': 5
    })
    
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_clusters': 5
    })

@dataclass
class ModelConfig:
    """Model-related configurations."""
    model_classes: Dict[str, Any] = field(default_factory=lambda: {
        'rf': RandomForestRegressor,
        'xgb': XGBRegressor,
        'lgbm': LGBMRegressor,
        'ada': AdaBoostRegressor,
        'catboost': CatBoostRegressor,
        'knn': KNeighborsRegressor
    })
    
    hyperparameter_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'xgb': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        },
        'lgbm': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 62, 124]
        },
        'ada': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'catboost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8]
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    })
    
    # Training parameters
    cv_splits: int = 5
    randomized_search_iterations: int = 10
    ensemble_cv_splits: int = 10
    ensemble_cv_shuffle: bool = True

@dataclass
class UIConfig:
    """UI-related configurations."""
    streamlit_theme: Dict[str, str] = field(default_factory=lambda: {
        'primaryColor': '#FF4B4B',
        'backgroundColor': '#FFFFFF',
        'secondaryBackgroundColor': '#F0F2F6',
        'textColor': '#262730',
        'font': 'sans serif'
    })
    
    app_name: str = 'ML Algo Trainer'
    app_icon: str = '🧠'
    
    # Visualization configurations
    max_rows_display: int = 100
    chart_height: int = 400
    chart_width: int = 600

@dataclass
class ModeConfig:
    """Mode-related configurations."""
    available_modes: List[str] = field(default_factory=lambda: [
        'Analysis', 'Training', 'Prediction'
    ])
    default_mode: str = 'Analysis'
    mode_transitions: Dict[str, List[str]] = field(default_factory=lambda: {
        'Analysis': ['Training'],
        'Training': ['Prediction'],
        'Prediction': ['Analysis', 'Training']
    })

class ConfigManager:
    """Central configuration management class."""
    def __init__(self):
        self.directories = DirectoryConfig()
        self.files = FileConfig()
        self.processing = ProcessingConfig()
        self.clustering = ClusteringConfig()
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.mode = ModeConfig()
        
        # Runtime configurations
        self.current_mode: str = self.mode.default_mode
        self.file_path: Optional[str] = None
        self.sheet_name: Optional[str] = None
        self.target_column: Optional[str] = None
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.unused_columns: List[str] = []
        self.all_columns: List[str] = []
        
        # Feature engineering flags
        self.use_polynomial_features: bool = True
        self.use_interaction_terms: bool = True
        self.use_statistical_features: bool = True
        
        # Clustering flags
        self.use_clustering: bool = False
        self.clustering_config: Dict[str, Dict] = {}
        self.clustering_2d_config: Dict[tuple, Dict] = {}
        self.clustering_2d_columns: List[str] = []
        
        # Training parameters
        self.train_size: float = 0.8
        self.models_to_use: List[str] = []
        self.tuning_method: str = 'None'
        
        # Initialize timestamps
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        self.all_columns = list(set(
            self.numerical_columns +
            self.categorical_columns +
            ([self.target_column] if self.target_column else []) +
            self.unused_columns
        ))
        self.last_updated = datetime.now()
    
    def set_mode(self, mode: str) -> None:
        """Set current mode if valid."""
        if mode in self.mode.available_modes:
            self.current_mode = mode
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_available_transitions(self) -> List[str]:
        """Get available mode transitions from current mode."""
        return self.mode.mode_transitions.get(self.current_mode, [])
    
    def validate(self) -> bool:
        """Validate current configuration."""
        try:
            assert self.current_mode in self.mode.available_modes
            assert all(isinstance(col, str) for col in self.all_columns)
            assert isinstance(self.train_size, float) and 0 < self.train_size < 1
            assert all(model in self.model.model_classes for model in self.models_to_use)
            return True
        except AssertionError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'directories': self.directories.__dict__,
            'files': self.files.__dict__,
            'processing': self.processing.__dict__,
            'clustering': self.clustering.__dict__,
            'model': self.model.__dict__,
            'ui': self.ui.__dict__,
            'mode': self.mode.__dict__,
            'runtime': {
                'current_mode': self.current_mode,
                'file_path': self.file_path,
                'sheet_name': self.sheet_name,
                'target_column': self.target_column,
                'numerical_columns': self.numerical_columns,
                'categorical_columns': self.categorical_columns,
                'unused_columns': self.unused_columns,
                'all_columns': self.all_columns,
                'use_polynomial_features': self.use_polynomial_features,
                'use_interaction_terms': self.use_interaction_terms,
                'use_statistical_features': self.use_statistical_features,
                'use_clustering': self.use_clustering,
                'clustering_config': self.clustering_config,
                'clustering_2d_config': self.clustering_2d_config,
                'clustering_2d_columns': self.clustering_2d_columns,
                'train_size': self.train_size,
                'models_to_use': self.models_to_use,
                'tuning_method': self.tuning_method
            },
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat()
            }
        }

# Create global configuration instance
config = ConfigManager()