import time
import functools
import logging
import psutil
import inspect
from typing import Any, Callable, Dict, Optional, Type
from datetime import datetime

from core.exceptions import (
    MLTrainerException,
    ValidationError,
    StateError
)
from core.state_manager import state_manager

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            performance_data = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_change': end_memory - start_memory,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Store performance data in state
            state_manager.set_state(
                f'metadata.performance.{func.__name__}.{datetime.now().isoformat()}',
                performance_data
            )
            
            return result
            
        except Exception as e:
            performance_data = {
                'function': func.__name__,
                'execution_time': time.time() - start_time,
                'memory_change': psutil.Process().memory_info().rss / 1024 / 1024 - start_memory,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            state_manager.set_state(
                f'metadata.performance.{func.__name__}.{datetime.now().isoformat()}',
                performance_data
            )
            
            raise
            
    return wrapper

def validate_input(validation_func: Optional[Callable] = None) -> Callable:
    """Decorator to validate function inputs."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Custom validation if provided
            if validation_func is not None:
                validation_result = validation_func(*args, **kwargs)
                if not validation_result:
                    raise ValidationError(
                        f"Input validation failed for {func.__name__}",
                        {'args': args, 'kwargs': kwargs}
                    )
            
            # Type checking based on annotations
            for param_name, param_value in bound_args.arguments.items():
                param_type = sig.parameters[param_name].annotation
                if param_type != inspect.Parameter.empty:
                    if not isinstance(param_value, param_type):
                        raise ValidationError(
                            f"Type validation failed for parameter {param_name}",
                            {
                                'expected_type': str(param_type),
                                'actual_type': str(type(param_value))
                            }
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution(logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator to log function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            func_name = func.__name__
            
            logger.info(f"Starting execution of {func_name}")
            logger.debug(f"Arguments: args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                logger.info(f"Completed {func_name} in {execution_time:.2f} seconds")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator

def handle_exceptions(
    error_class: Type[Exception] = MLTrainerException,
    message: Optional[str] = None
) -> Callable:
    """Decorator to handle exceptions with custom error class."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, error_class):
                    raise
                    
                error_msg = message or f"Error in {func.__name__}: {str(e)}"
                raise error_class(error_msg, details={
                    'original_error': str(e),
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }) from e
                
        return wrapper
    return decorator

def require_state(state_path: str) -> Callable:
    """Decorator to ensure required state exists."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                state = state_manager.get_state(state_path)
                if state is None:
                    raise StateError(
                        f"Required state not found: {state_path}",
                        {'path': state_path}
                    )
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, StateError):
                    raise
                raise StateError(
                    f"Error accessing state: {state_path}",
                    {'path': state_path, 'error': str(e)}
                ) from e
        return wrapper
    return decorator

def cache_result(
    cache_path: str,
    ttl: Optional[int] = None
) -> Callable:
    """Decorator to cache function results in state."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{cache_path}.{func.__name__}"
            
            # Try to get from cache
            try:
                cached = state_manager.get_state(cache_key)
                if cached is not None:
                    cache_time = cached.get('timestamp')
                    if cache_time:
                        cache_age = (datetime.now() - datetime.fromisoformat(cache_time)).total_seconds()
                        if ttl is None or cache_age < ttl:
                            return cached['result']
            except StateError:
                pass
            
            # Calculate and cache result
            result = func(*args, **kwargs)
            state_manager.set_state(cache_key, {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'args': args,
                'kwargs': kwargs
            })
            
            return result
            
        return wrapper
    return decorator

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator to retry function execution on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    
            raise last_exception
            
        return wrapper
    return decorator

def deprecate(
    message: Optional[str] = None,
    alternative: Optional[str] = None
) -> Callable:
    """Decorator to mark functions as deprecated."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_message = message or f"{func.__name__} is deprecated."
            if alternative:
                warning_message += f" Use {alternative} instead."
                
            logging.warning(warning_message)
            return func(*args, **kwargs)
            
        return wrapper
    return decorator