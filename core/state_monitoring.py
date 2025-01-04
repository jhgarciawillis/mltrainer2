import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import psutil
import numpy as np
from pathlib import Path
import json
import threading
import queue

from core.config import config
from core.exceptions import StateError
from utils.loggers import logger
from utils.decorators import monitor_performance

class StateMonitor:
    """Monitor and track application state changes and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.state_changes: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }
        self.error_counts: Dict[str, int] = {}
        self.active_operations: Dict[str, datetime] = {}
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_queue: queue.Queue = queue.Queue()
        self.should_monitor = False
        
        # Initialize monitoring
        self.start_monitoring()
    
    @monitor_performance
    def start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.should_monitor = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("State monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring thread."""
        self.should_monitor = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            logger.info("State monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.should_monitor:
            try:
                # Collect current metrics
                self._collect_metrics()
                
                # Process any queued operations
                while not self.monitor_queue.empty():
                    operation = self.monitor_queue.get_nowait()
                    self._process_operation(operation)
                
                # Check for long-running operations
                self._check_long_running_operations()
                
                # Sleep for monitoring interval
                time.sleep(config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    @monitor_performance
    def _collect_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_metrics['memory_usage'].append(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.performance_metrics['disk_usage'].append(disk.percent)
            
            # Trim metrics lists if too long
            max_metrics = config.max_metrics_history
            for metric_list in self.performance_metrics.values():
                if len(metric_list) > max_metrics:
                    metric_list.pop(0)
                    
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    def _process_operation(self, operation: Dict[str, Any]) -> None:
        """Process monitored operation."""
        op_type = operation.get('type')
        if op_type == 'state_change':
            self.state_changes.append(operation)
        elif op_type == 'error':
            error_type = operation.get('error_type', 'unknown')
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    @monitor_performance
    def record_state_change(
        self,
        component: str,
        change_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a state change."""
        change = {
            'type': 'state_change',
            'component': component,
            'change_type': change_type,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.monitor_queue.put(change)
        logger.debug(f"State change recorded: {change}")
    
    @monitor_performance
    def record_error(
        self,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an error occurrence."""
        error = {
            'type': 'error',
            'error_type': error_type,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.monitor_queue.put(error)
        logger.error(f"Error recorded: {error}")
    
    @monitor_performance
    def start_operation(self, operation_name: str) -> None:
        """Record start of an operation."""
        self.active_operations[operation_name] = datetime.now()
        logger.debug(f"Operation started: {operation_name}")
    
    @monitor_performance
    def end_operation(self, operation_name: str) -> None:
        """Record end of an operation."""
        if operation_name in self.active_operations:
            start_time = self.active_operations.pop(operation_name)
            duration = datetime.now() - start_time
            logger.debug(f"Operation ended: {operation_name}, duration: {duration}")
    
    def _check_long_running_operations(self) -> None:
        """Check for operations that might be running too long."""
        current_time = datetime.now()
        timeout = timedelta(seconds=config.operation_timeout)
        
        for op_name, start_time in list(self.active_operations.items()):
            if current_time - start_time > timeout:
                logger.warning(f"Long-running operation detected: {op_name}")
    
    @monitor_performance
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'avg': np.mean(values),
                    'max': max(values),
                    'min': min(values)
                }
        return summary
    
    @monitor_performance
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts.copy(),
            'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }
    
    @monitor_performance
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'uptime': datetime.now() - self.start_time,
            'metrics': self.get_metrics_summary(),
            'errors': self.get_error_summary(),
            'state_changes': len(self.state_changes),
            'active_operations': len(self.active_operations)
        }
    
    @monitor_performance
    def save_monitoring_data(self, path: Optional[Path] = None) -> None:
        """Save monitoring data to disk."""
        if path is None:
            path = config.directories.base_dir / 'monitoring'
        
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        metrics_file = path / f'metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=4)
        
        # Save state changes
        changes_file = path / f'state_changes_{timestamp}.json'
        with open(changes_file, 'w') as f:
            json.dump(self.state_changes, f, indent=4)
        
        # Save error counts
        errors_file = path / f'errors_{timestamp}.json'
        with open(errors_file, 'w') as f:
            json.dump(self.error_counts, f, indent=4)
        
        logger.info(f"Monitoring data saved to {path}")

# Create global state monitor instance
state_monitor = StateMonitor()