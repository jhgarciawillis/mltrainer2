import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import yaml
from io import StringIO, BytesIO

from core.config import config
from core.exceptions import DataLoadError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.validators import validate_dataframe, validate_file_path
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataLoader:
    """Handle all data loading operations with validation and preprocessing."""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.parquet': self._load_parquet,
            '.json': self._load_json,
            '.yaml': self._load_yaml,
            '.pkl': self._load_pickle
        }
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        self.data_info: Dict[str, Dict[str, Any]] = {}
    
    @monitor_performance
    @handle_exceptions(DataLoadError)
    def load_file(
        self,
        file_path: Union[str, Path],
        sheet_name: Optional[str] = None,
        validate: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from file with automatic format detection."""
        path = Path(file_path)
        
        # Validate file path
        validate_file_path(
            path,
            allowed_extensions=set(self.supported_formats.keys()),
            must_exist=True
        )
        
        # Load data based on file extension
        file_extension = path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise DataLoadError(
                f"Unsupported file format: {file_extension}",
                details={'supported_formats': list(self.supported_formats.keys())}
            )
        
        data = self.supported_formats[file_extension](path, sheet_name, **kwargs)
        
        # Validate loaded data if requested
        if validate:
            validate_dataframe(data)
        
        # Store loaded data info
        data_id = str(path)
        self.loaded_data[data_id] = data
        self.data_info[data_id] = self._get_data_info(data, path)
        
        # Update state
        state_manager.set_state(
            f'data.loaded_files.{path.stem}',
            self.data_info[data_id]
        )
        
        logger.info(f"Successfully loaded data from {path}")
        return data
    
    @log_execution()
    def _load_csv(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            # Set default encoding and handle common encoding issues
            kwargs.setdefault('encoding', 'utf-8')
            try:
                return pd.read_csv(path, **kwargs)
            except UnicodeDecodeError:
                kwargs['encoding'] = 'latin1'
                return pd.read_csv(path, **kwargs)
        except Exception as e:
            raise DataLoadError(
                f"Error loading CSV file: {str(e)}",
                details={'path': str(path)}
            ) from e
    
    @log_execution()
    def _load_excel(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            if sheet_name is None:
                # Get list of sheets
                xls = pd.ExcelFile(path)
                if len(xls.sheet_names) == 1:
                    sheet_name = xls.sheet_names[0]
                else:
                    raise DataLoadError(
                        "Multiple sheets found, please specify sheet_name",
                        details={'available_sheets': xls.sheet_names}
                    )
            
            return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
            
        except Exception as e:
            raise DataLoadError(
                f"Error loading Excel file: {str(e)}",
                details={'path': str(path), 'sheet_name': sheet_name}
            ) from e
    
    @log_execution()
    def _load_parquet(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as e:
            raise DataLoadError(
                f"Error loading Parquet file: {str(e)}",
                details={'path': str(path)}
            ) from e
    
    @log_execution()
    def _load_json(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            return pd.read_json(path, **kwargs)
        except Exception as e:
            raise DataLoadError(
                f"Error loading JSON file: {str(e)}",
                details={'path': str(path)}
            ) from e
    
    @log_execution()
    def _load_yaml(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from YAML file."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return pd.DataFrame(data)
        except Exception as e:
            raise DataLoadError(
                f"Error loading YAML file: {str(e)}",
                details={'path': str(path)}
            ) from e
    
    @log_execution()
    def _load_pickle(
        self,
        path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Pickle file."""
        try:
            return pd.read_pickle(path, **kwargs)
        except Exception as e:
            raise DataLoadError(
                f"Error loading Pickle file: {str(e)}",
                details={'path': str(path)}
            ) from e
    
    @monitor_performance
    def load_streaming(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 10000,
        **kwargs
    ) -> pd.DataFrame:
        """Load large files in chunks."""
        path = Path(file_path)
        chunks = []
        
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_size, **kwargs):
                chunks.append(chunk)
                
            return pd.concat(chunks, ignore_index=True)
            
        except Exception as e:
            raise DataLoadError(
                f"Error in streaming load: {str(e)}",
                details={'path': str(path), 'chunk_size': chunk_size}
            ) from e
    
    @monitor_performance
    def load_multiple_files(
        self,
        file_patterns: List[Union[str, Path]],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple files matching patterns."""
        loaded_files = {}
        
        for pattern in file_patterns:
            for file_path in Path().glob(str(pattern)):
                try:
                    loaded_files[str(file_path)] = self.load_file(
                        file_path,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        
        return loaded_files
    
    def _get_data_info(
        self,
        data: pd.DataFrame,
        path: Path
    ) -> Dict[str, Any]:
        """Get information about loaded data."""
        return {
            'path': str(path),
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'loaded_at': pd.Timestamp.now().isoformat()
        }
    
    @monitor_performance
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        return {
            'total_files': len(self.loaded_data),
            'total_rows': sum(df.shape[0] for df in self.loaded_data.values()),
            'total_columns': sum(df.shape[1] for df in self.loaded_data.values()),
            'file_info': self.data_info
        }

# Create global data loader instance
data_loader = DataLoader()