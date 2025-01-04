import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from core.config import config
from core.exceptions import StateError

class StateManager:
    """Manages application state across different modes."""
    
    def __init__(self):
        self.state: Dict[str, Any] = {
            'mode': config.current_mode,
            'data': {
                'original_data': None,
                'cleaned_data': None,
                'preprocessed_data': None,
                'feature_engineered_data': None
            },
            'analysis': {
                'results': None,
                'visualizations': {},
                'reports': {},
                'metadata': {}
            },
            'training': {
                'models': {},
                'results': {},
                'history': {},
                'metadata': {}
            },
            'prediction': {
                'results': None,
                'evaluations': {},
                'metadata': {}
            },
            'preprocessing': {
                'scalers': {},
                'encoders': {},
                'imputers': {}
            },
            'clustering': {
                'models': {},
                'results': {},
                'metadata': {}
            },
            'feature_engineering': {
                'selected_features': [],
                'feature_importance': {},
                'metadata': {}
            },
            'ui': {
                'current_page': 'main',
                'sidebar_state': 'expanded',
                'user_preferences': {}
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        self.state_history: List[Dict[str, Any]] = []
        self.max_history_size: int = 10
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for state management."""
        logger = logging.getLogger('state_manager')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(config.directories.base_dir / 'state_manager.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def get_state(self, path: Optional[str] = None) -> Any:
        """Get state or part of state using dot notation path."""
        if path is None:
            return self.state
            
        try:
            current = self.state
            for key in path.split('.'):
                current = current[key]
            return current
        except KeyError as e:
            self.logger.error(f"State path not found: {path}")
            raise StateError(f"Invalid state path: {path}", {'path': path}) from e

    def set_state(self, path: str, value: Any, save_history: bool = True) -> None:
        """Set state using dot notation path."""
        try:
            if save_history:
                self._save_to_history()
                
            keys = path.split('.')
            current = self.state
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
            
            self.state['metadata']['last_updated'] = datetime.now().isoformat()
            self.logger.info(f"State updated: {path}")
            
        except Exception as e:
            self.logger.error(f"Error setting state: {path}")
            raise StateError(f"Error setting state: {path}", {'path': path, 'value': value}) from e

    def _save_to_history(self) -> None:
        """Save current state to history."""
        self.state_history.append(self.state.copy())
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)

    def undo(self) -> bool:
        """Revert to previous state."""
        try:
            if self.state_history:
                self.state = self.state_history.pop()
                self.logger.info("State reverted to previous version")
                return True
            return False
        except Exception as e:
            self.logger.error("Error reverting state")
            raise StateError("Error reverting state") from e

    def save_state(self, path: Optional[Path] = None) -> None:
        """Save current state to disk."""
        try:
            if path is None:
                path = self._get_state_path()
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save different components in different formats
            state_to_save = {
                'metadata': self.state['metadata'],
                'ui': self.state['ui'],
                'feature_engineering': self.state['feature_engineering']
            }
            
            # Save main state as JSON
            with open(path, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            
            # Save data components using joblib
            data_path = path.parent / 'data_state.joblib'
            joblib.dump(self.state['data'], data_path)
            
            # Save models using joblib
            models_path = path.parent / 'models_state.joblib'
            joblib.dump(self.state['training']['models'], models_path)
            
            self.logger.info(f"State saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise StateError("Error saving state", {'path': str(path)}) from e

    def load_state(self, path: Optional[Path] = None) -> None:
        """Load state from disk."""
        try:
            if path is None:
                path = self._get_state_path()
            
            if not path.exists():
                self.logger.warning(f"No state file found at {path}")
                return
            
            # Load main state from JSON
            with open(path, 'r') as f:
                loaded_state = json.load(f)
            
            # Load data components
            data_path = path.parent / 'data_state.joblib'
            if data_path.exists():
                self.state['data'] = joblib.load(data_path)
            
            # Load models
            models_path = path.parent / 'models_state.joblib'
            if models_path.exists():
                self.state['training']['models'] = joblib.load(models_path)
            
            # Update state with loaded components
            self.state.update(loaded_state)
            self.logger.info(f"State loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            raise StateError("Error loading state", {'path': str(path)}) from e

    def _get_state_path(self) -> Path:
        """Get appropriate state path based on current mode."""
        mode_paths = {
            'Analysis': config.directories.analysis_state,
            'Training': config.directories.training_state,
            'Prediction': config.directories.prediction_dir
        }
        
        base_path = mode_paths.get(self.state['mode'])
        if base_path is None:
            raise StateError(f"Invalid mode for state path: {self.state['mode']}")
            
        return base_path / 'state.json'

    def clear_state(self, paths: Optional[List[str]] = None) -> None:
        """Clear specific paths or entire state."""
        try:
            if paths is None:
                # Reset to initial state but keep metadata
                metadata = self.state['metadata'].copy()
                self.__init__()
                self.state['metadata'] = metadata
                self.state['metadata']['last_updated'] = datetime.now().isoformat()
            else:
                for path in paths:
                    self.set_state(path, None, save_history=True)
                    
            self.logger.info("State cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing state: {str(e)}")
            raise StateError("Error clearing state", {'paths': paths}) from e

    def validate_state(self) -> bool:
        """Validate current state."""
        try:
            required_keys = ['mode', 'data', 'metadata']
            if not all(key in self.state for key in required_keys):
                return False
                
            if not isinstance(self.state['metadata']['created_at'], str):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"State validation failed: {str(e)}")
            return False

# Create global state manager instance
state_manager = StateManager()