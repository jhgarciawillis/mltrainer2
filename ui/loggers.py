import logging
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from core.config import config
from core.exceptions import MLTrainerException

class CustomFormatter(logging.Formatter):
    """Custom formatter with color support and extended information."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, include_path: bool = False):
        self.include_path = include_path
        super().__init__(
            fmt=self._get_format(),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_format(self) -> str:
        """Get format string based on configuration."""
        base_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        if self.include_path:
            base_fmt = f'{base_fmt} (%(pathname)s:%(lineno)d)'
        return base_fmt
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and additional information."""
        # Save original values
        original_msg = record.msg
        original_levelname = record.levelname
        
        # Add color if outputting to terminal
        if sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add extra context if available
        if hasattr(record, 'extra_context'):
            record.msg = f"{original_msg} | Context: {record.extra_context}"
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original values
        record.msg = original_msg
        record.levelname = original_levelname
        
        return formatted

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra context if available
        if hasattr(record, 'extra_context'):
            log_data['context'] = record.extra_context
        
        return json.dumps(log_data)

class MLTrainerLogger:
    """Central logger for ML Trainer application."""
    
    def __init__(
        self,
        name: str = 'mltrainer',
        level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        include_json: bool = True
    ):
        self.name = name
        self.level = level
        self.log_dir = log_dir or config.directories.base_dir / 'logs'
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.include_json = include_json
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(CustomFormatter(include_path=True))
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(CustomFormatter(include_path=True))
        self.logger.addHandler(file_handler)
        
        # JSON file handler if enabled
        if self.include_json:
            json_file = self.log_dir / f"{self.name}.json.log"
            json_handler = RotatingFileHandler(
                json_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            json_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(json_handler)
        
        # Daily rotating handler for archival
        daily_file = self.log_dir / f"{self.name}.daily.log"
        daily_handler = TimedRotatingFileHandler(
            daily_file,
            when='midnight',
            interval=1,
            backupCount=30
        )
        daily_handler.setFormatter(CustomFormatter(include_path=False))
        self.logger.addHandler(daily_handler)
    
    def _log_with_context(
        self,
        level: int,
        msg: str,
        extra_context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> None:
        """Log message with additional context."""
        extra = {'extra_context': extra_context} if extra_context else None
        self.logger.log(level, msg, extra=extra, exc_info=exc_info)
    
    def debug(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, msg, context)
    
    def info(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, msg, context)
    
    def warning(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, msg, context)
    
    def error(
        self,
        msg: str,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, msg, context, error)
    
    def critical(
        self,
        msg: str,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, msg, context, error)
    
    def exception(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, msg, context, sys.exc_info())
    
    def log_ml_trainer_exception(
        self,
        exception: MLTrainerException
    ) -> None:
        """Log MLTrainerException with all details."""
        context = {
            'error_code': exception.code,
            'details': exception.details,
            'timestamp': exception.timestamp.isoformat()
        }
        if exception.previous_exception:
            context['previous_exception'] = str(exception.previous_exception)
        
        self.error(str(exception), exception, context)
    
    def get_logs(
        self,
        level: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve logs with optional filtering."""
        json_log_file = self.log_dir / f"{self.name}.json.log"
        if not json_log_file.exists():
            return []
        
        logs = []
        with open(json_log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    
                    # Apply filters
                    if level and log_entry['level'] != logging.getLevelName(level):
                        continue
                        
                    log_time = datetime.strptime(
                        log_entry['timestamp'],
                        '%Y-%m-%d %H:%M:%S'
                    )
                    
                    if start_time and log_time < start_time:
                        continue
                    if end_time and log_time > end_time:
                        continue
                        
                    logs.append(log_entry)
                    
                except json.JSONDecodeError:
                    continue
                    
                if limit and len(logs) >= limit:
                    break
        
        return logs

# Create global logger instance
logger = MLTrainerLogger()