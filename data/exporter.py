import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
import json
import yaml
import pickle
import xlsxwriter
from datetime import datetime
import zipfile
import io
import csv

from core.config import config
from core.exceptions import ExportError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataExporter:
    """Handle all data export operations."""
    
    def __init__(self):
        self.export_history: List[Dict[str, Any]] = []
        self.supported_formats: Set[str] = {
            'csv', 'excel', 'json', 'yaml', 'parquet', 'pickle', 'html'
        }
        self.compression_options: Set[str] = {'zip', 'gzip', 'bz2', 'xz'}
    
    @monitor_performance
    @handle_exceptions(ExportError)
    def export_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        file_path: Union[str, Path],
        format: str = 'csv',
        compression: Optional[str] = None,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export data to specified format."""
        path = Path(file_path)
        format = format.lower()
        
        if format not in self.supported_formats:
            raise ExportError(
                f"Unsupported format: {format}",
                details={'supported_formats': list(self.supported_formats)}
            )
        
        if compression and compression not in self.compression_options:
            raise ExportError(
                f"Unsupported compression: {compression}",
                details={'supported_compression': list(self.compression_options)}
            )
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Handle multiple DataFrames
            if isinstance(data, dict):
                return self._export_multiple_dataframes(
                    data, path, format, compression, **kwargs
                )
            
            # Handle single DataFrame
            return self._export_single_dataframe(
                data, path, format, compression, sheet_name, **kwargs
            )
            
        except Exception as e:
            raise ExportError(
                f"Error exporting data: {str(e)}",
                details={
                    'format': format,
                    'compression': compression,
                    'path': str(path)
                }
            ) from e
    
    def _export_single_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
        format: str,
        compression: Optional[str],
        sheet_name: Optional[str],
        **kwargs
    ) -> Path:
        """Export single DataFrame."""
        export_func = getattr(self, f'_export_to_{format}')
        exported_path = export_func(
            df, path, compression, sheet_name, **kwargs
        )
        
        self._record_export(
            path=exported_path,
            format=format,
            compression=compression,
            rows=len(df),
            columns=len(df.columns)
        )
        
        return exported_path
    
    def _export_multiple_dataframes(
        self,
        data_dict: Dict[str, pd.DataFrame],
        path: Path,
        format: str,
        compression: Optional[str],
        **kwargs
    ) -> Path:
        """Export multiple DataFrames."""
        if format == 'excel':
            return self._export_to_excel_multiple(
                data_dict, path, compression, **kwargs
            )
        
        # For other formats, create a directory and save each DataFrame
        base_path = path.with_suffix('')
        base_path.mkdir(parents=True, exist_ok=True)
        
        exported_paths = []
        for name, df in data_dict.items():
            file_path = base_path / f"{name}.{format}"
            exported_path = self._export_single_dataframe(
                df, file_path, format, compression, name, **kwargs
            )
            exported_paths.append(exported_path)
        
        # Create zip file if requested
        if compression == 'zip':
            zip_path = path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for p in exported_paths:
                    zf.write(p, p.name)
                    p.unlink()  # Remove individual files
            base_path.rmdir()
            return zip_path
        
        return base_path
    
    def _export_to_csv(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to CSV format."""
        export_path = path.with_suffix('.csv')
        if compression:
            export_path = export_path.with_suffix(f'.csv.{compression}')
        
        df.to_csv(export_path, index=False, compression=compression, **kwargs)
        return export_path
    
    def _export_to_excel(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to Excel format."""
        export_path = path.with_suffix('.xlsx')
        sheet_name = sheet_name or 'Sheet1'
        
        with pd.ExcelWriter(
            export_path,
            engine='xlsxwriter',
            options={'remove_timezone': True}
        ) as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                **kwargs
            )
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.set_column(i, i, max_length + 2)
        
        return export_path
    
    def _export_to_excel_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        path: Path,
        compression: Optional[str],
        **kwargs
    ) -> Path:
        """Export multiple DataFrames to Excel."""
        export_path = path.with_suffix('.xlsx')
        
        with pd.ExcelWriter(
            export_path,
            engine='xlsxwriter',
            options={'remove_timezone': True}
        ) as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],  # Excel sheet name length limit
                    index=False,
                    **kwargs
                )
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name[:31]]
                for i, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    )
                    worksheet.set_column(i, i, max_length + 2)
        
        return export_path
    
    def _export_to_json(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to JSON format."""
        export_path = path.with_suffix('.json')
        if compression:
            export_path = export_path.with_suffix(f'.json.{compression}')
        
        df.to_json(export_path, compression=compression, **kwargs)
        return export_path
    
    def _export_to_yaml(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to YAML format."""
        export_path = path.with_suffix('.yaml')
        
        # Convert DataFrame to dict
        data_dict = df.to_dict(orient='records')
        
        with open(export_path, 'w') as f:
            yaml.dump(data_dict, f, **kwargs)
        
        return export_path
    
    def _export_to_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to Parquet format."""
        export_path = path.with_suffix('.parquet')
        
        df.to_parquet(export_path, compression=compression or 'snappy', **kwargs)
        return export_path
    
    def _export_to_pickle(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to Pickle format."""
        export_path = path.with_suffix('.pkl')
        if compression:
            export_path = export_path.with_suffix(f'.pkl.{compression}')
        
        df.to_pickle(export_path, compression=compression, **kwargs)
        return export_path
    
    def _export_to_html(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: Optional[str],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Export to HTML format."""
        export_path = path.with_suffix('.html')
        
        df.to_html(
            export_path,
            index=False,
            table_id='data_table',
            classes=['table', 'table-striped', 'table-hover'],
            **kwargs
        )
        
        return export_path
    
    def _record_export(self, **kwargs) -> None:
        """Record export operation."""
        export_record = {
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.export_history.append(export_record)
        
        # Update state
        state_manager.set_state(
            f'data.exports.{len(self.export_history)}',
            export_record
        )
    
    @monitor_performance
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of export operations."""
        return {
            'total_exports': len(self.export_history),
            'formats_used': list(set(
                record['format'] for record in self.export_history
            )),
            'total_rows_exported': sum(
                record.get('rows', 0) for record in self.export_history
            ),
            'recent_exports': self.export_history[-5:]
        }
    
    @monitor_performance
    def save_export_history(
        self,
        path: Optional[Path] = None
    ) -> None:
        """Save export history to disk."""
        if path is None:
            path = config.directories.base_dir / 'export_history'
        
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        history_file = path / f'export_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            json.dump(self.export_history, f, indent=4)
        
        logger.info(f"Export history saved to {history_file}")

# Create global exporter instance
exporter = DataExporter()