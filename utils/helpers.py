import os
import shutil
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from core.exceptions import MLTrainerException
from core.config import config
from core.state_manager import state_manager
from utils.decorators import log_execution, handle_exceptions, monitor_performance
from utils.validators import validate_dataframe, validate_file_path

@log_execution()
@handle_exceptions(MLTrainerException)
def setup_directory(directory_path: Path) -> None:
    """Create or clean directory."""
    if directory_path.exists():
        shutil.rmtree(directory_path)
    directory_path.mkdir(parents=True)

@log_execution()
def setup_logging(
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = 1024*1024,
    backup_count: int = 5
) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('mltrainer')
    logger.setLevel(level)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@monitor_performance
def create_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Create formatted timestamp string."""
    return datetime.now().strftime(format_str)

@monitor_performance
def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.'
) -> Dict[str, Any]:
    """Flatten nested dictionary with custom separator."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@monitor_performance
@handle_exceptions(MLTrainerException)
def save_artifact(
    artifact: Any,
    path: Path,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save artifact with metadata."""
    try:
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save artifact based on type
        if artifact_type == 'dataframe':
            if path.suffix == '.csv':
                artifact.to_csv(path, index=False)
            else:
                artifact.to_parquet(path)
        elif artifact_type == 'figure':
            if path.suffix == '.html':
                artifact.write_html(path)
            else:
                artifact.write_image(path)
        elif artifact_type == 'model':
            import joblib
            joblib.dump(artifact, path)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = path.with_suffix('.meta.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        # Update state
        state_manager.set_state(
            f'metadata.artifacts.{path.stem}',
            {
                'path': str(path),
                'type': artifact_type,
                'saved_at': create_timestamp(),
                'metadata': metadata
            }
        )
            
    except Exception as e:
        raise MLTrainerException(
            f"Error saving artifact: {str(e)}",
            details={'path': str(path), 'type': artifact_type}
        ) from e

@monitor_performance
@handle_exceptions(MLTrainerException)
def load_artifact(
    path: Path,
    artifact_type: str,
    validate: bool = True
) -> Any:
    """Load artifact with optional validation."""
    try:
        # Validate path
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        # Load artifact based on type
        if artifact_type == 'dataframe':
            if path.suffix == '.csv':
                artifact = pd.read_csv(path)
            else:
                artifact = pd.read_parquet(path)
            if validate:
                validate_dataframe(artifact)
        elif artifact_type == 'figure':
            if path.suffix == '.html':
                artifact = go.Figure(px.load_figure(path))
            else:
                raise ValueError("Can only load HTML figures")
        elif artifact_type == 'model':
            import joblib
            artifact = joblib.load(path)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        # Load metadata if exists
        metadata_path = path.with_suffix('.meta.json')
        metadata = None
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Update state
        state_manager.set_state(
            f'metadata.artifacts.{path.stem}',
            {
                'path': str(path),
                'type': artifact_type,
                'loaded_at': create_timestamp(),
                'metadata': metadata
            }
        )
        
        return artifact
        
    except Exception as e:
        raise MLTrainerException(
            f"Error loading artifact: {str(e)}",
            details={'path': str(path), 'type': artifact_type}
        ) from e

@monitor_performance
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

@monitor_performance
def check_resource_availability() -> Dict[str, Any]:
    """Check system resource availability."""
    import psutil
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_available': psutil.virtual_memory().available / (1024 * 1024),  # MB
        'disk_space_available': psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # GB
    }

@monitor_performance
def identify_high_correlation_features(
    data: pd.DataFrame,
    threshold: float = 0.9
) -> List[Tuple[str, str]]:
    """Identify highly correlated feature pairs."""
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    return [
        (col1, col2)
        for col1, col2 in zip(*np.where(upper > threshold))
    ]

@monitor_performance
def create_backup(
    file_path: Path,
    max_backups: int = 5
) -> Path:
    """Create backup with rotation."""
    backup_base = file_path.with_suffix(file_path.suffix + '.backup')
    
    # Rotate existing backups
    existing_backups = sorted(
        file_path.parent.glob(f"{file_path.stem}*.backup"),
        reverse=True
    )
    
    # Remove oldest backup if limit reached
    while len(existing_backups) >= max_backups:
        existing_backups[-1].unlink()
        existing_backups.pop()
    
    # Create new backup with timestamp
    timestamp = create_timestamp()
    backup_path = file_path.with_suffix(f".{timestamp}.backup")
    shutil.copy2(file_path, backup_path)
    
    return backup_path

@monitor_performance
@handle_exceptions(MLTrainerException)
def restore_backup(
    backup_path: Path,
    target_path: Path,
    validate: bool = True
) -> None:
    """Restore from backup with validation."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    
    if target_path.exists():
        # Create temporary backup of current file
        temp_backup = create_backup(target_path)
        
        try:
            # Restore from backup
            shutil.copy2(backup_path, target_path)
            
            # Validate restored file if requested
            if validate:
                validate_file_path(target_path, must_exist=True)
                
            # Remove temporary backup
            temp_backup.unlink()
            
        except Exception as e:
            # Restore from temporary backup on error
            shutil.copy2(temp_backup, target_path)
            temp_backup.unlink()
            raise MLTrainerException(
                f"Error restoring backup: {str(e)}",
                details={'backup_path': str(backup_path)}
            ) from e

@monitor_performance
def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get detailed file information."""
    stat = file_path.stat()
    return {
        'size': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'accessed': datetime.fromtimestamp(stat.st_atime),
        'is_file': file_path.is_file(),
        'extension': file_path.suffix,
        'permissions': oct(stat.st_mode)[-3:]
    }

@monitor_performance
def format_size(
    size_bytes: int,
    precision: int = 2
) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.{precision}f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.{precision}f} PB"