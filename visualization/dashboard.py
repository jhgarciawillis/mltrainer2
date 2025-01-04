import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import config
from core.exceptions import VisualizationError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization.plotter import plotter
from visualization.styler import style_manager

class DashboardManager:
    """Handle dashboard creation and management."""
    
    def __init__(self):
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.layouts: Dict[str, Dict[str, Any]] = {}
        self.current_dashboard: Optional[str] = None
        self.dashboard_history: List[Dict[str, Any]] = []
    
    @monitor_performance
    @handle_exceptions(VisualizationError)
    def create_dashboard(
        self,
        name: str,
        title: str,
        layout: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create new dashboard."""
        if name in self.dashboards:
            raise VisualizationError(f"Dashboard already exists: {name}")
        
        dashboard = {
            'title': title,
            'layout': layout or self._get_default_layout(),
            'plots': {},
            'filters': {},
            'metadata': {
                'created_at': pd.Timestamp.now().isoformat(),
                'last_updated': pd.Timestamp.now().isoformat()
            }
        }
        
        self.dashboards[name] = dashboard
        self.current_dashboard = name
        
        self._record_dashboard_action('creation', name)
    
    @monitor_performance
    def add_plot(
        self,
        plot_name: str,
        plot_type: str,
        data: Any,
        position: Optional[Dict[str, int]] = None,
        **plot_kwargs
    ) -> None:
        """Add plot to current dashboard."""
        if self.current_dashboard is None:
            raise VisualizationError("No active dashboard")
        
        # Create plot
        fig = plotter.create_plot(plot_type, data, **plot_kwargs)
        
        # Apply current theme
        fig = style_manager.apply_theme_to_figure(fig)
        
        # Add to dashboard
        dashboard = self.dashboards[self.current_dashboard]
        dashboard['plots'][plot_name] = {
            'figure': fig,
            'position': position or self._get_next_position(),
            'type': plot_type,
            'kwargs': plot_kwargs
        }
        
        dashboard['metadata']['last_updated'] = pd.Timestamp.now().isoformat()
        
        self._record_dashboard_action('plot_addition', plot_name)
    
    @monitor_performance
    def add_filter(
        self,
        filter_name: str,
        filter_type: str,
        options: List[Any],
        default: Optional[Any] = None,
        position: Optional[Dict[str, int]] = None
    ) -> None:
        """Add filter to current dashboard."""
        if self.current_dashboard is None:
            raise VisualizationError("No active dashboard")
        
        dashboard = self.dashboards[self.current_dashboard]
        dashboard['filters'][filter_name] = {
            'type': filter_type,
            'options': options,
            'default': default,
            'position': position or self._get_next_filter_position(),
            'value': default
        }
        
        dashboard['metadata']['last_updated'] = pd.Timestamp.now().isoformat()
        
        self._record_dashboard_action('filter_addition', filter_name)
    
    @monitor_performance
    def update_plot(
        self,
        plot_name: str,
        data: Any,
        **plot_kwargs
    ) -> None:
        """Update existing plot in dashboard."""
        if self.current_dashboard is None:
            raise VisualizationError("No active dashboard")
        
        dashboard = self.dashboards[self.current_dashboard]
        if plot_name not in dashboard['plots']:
            raise VisualizationError(f"Plot not found: {plot_name}")
        
        plot_info = dashboard['plots'][plot_name]
        
        # Create updated plot
        fig = plotter.create_plot(
            plot_info['type'],
            data,
            **(plot_kwargs or plot_info['kwargs'])
        )
        
        # Apply theme
        fig = style_manager.apply_theme_to_figure(fig)
        
        # Update dashboard
        plot_info['figure'] = fig
        if plot_kwargs:
            plot_info['kwargs'].update(plot_kwargs)
        
        dashboard['metadata']['last_updated'] = pd.Timestamp.now().isoformat()
        
        self._record_dashboard_action('plot_update', plot_name)
    
    @monitor_performance
    def render_dashboard(
        self,
        dashboard_name: Optional[str] = None
    ) -> None:
        """Render dashboard in Streamlit."""
        name = dashboard_name or self.current_dashboard
        if name is None:
            raise VisualizationError("No dashboard specified")
        
        dashboard = self.dashboards[name]
        
        # Display title
        st.title(dashboard['title'])
        
        # Create layout
        layout = dashboard['layout']
        num_rows = layout.get('rows', 2)
        num_cols = layout.get('columns', 2)
        
        # Render filters
        if dashboard['filters']:
            st.sidebar.title("Filters")
            filter_values = self._render_filters(dashboard['filters'])
        else:
            filter_values = {}
        
        # Create grid layout
        grid = []
        for i in range(num_rows):
            cols = st.columns(num_cols)
            grid.append(cols)
        
        # Render plots in grid
        for plot_name, plot_info in dashboard['plots'].items():
            position = plot_info['position']
            row, col = position.get('row', 0), position.get('column', 0)
            
            if 0 <= row < num_rows and 0 <= col < num_cols:
                with grid[row][col]:
                    st.plotly_chart(
                        plot_info['figure'],
                        use_container_width=True
                    )
        
        self._record_dashboard_action('render', name)
    
    def _render_filters(
        self,
        filters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Render dashboard filters."""
        filter_values = {}
        
        for name, filter_info in filters.items():
            filter_type = filter_info['type']
            options = filter_info['options']
            default = filter_info['default']
            
            if filter_type == 'select':
                value = st.sidebar.selectbox(
                    name,
                    options=options,
                    index=options.index(default) if default in options else 0
                )
            elif filter_type == 'multiselect':
                value = st.sidebar.multiselect(
                    name,
                    options=options,
                    default=default or []
                )
            elif filter_type == 'slider':
                value = st.sidebar.slider(
                    name,
                    min_value=options[0],
                    max_value=options[1],
                    value=default or options[0]
                )
            else:
                continue
            
            filter_values[name] = value
            filters[name]['value'] = value
        
        return filter_values
    
    def _get_default_layout(self) -> Dict[str, Any]:
        """Get default dashboard layout."""
        return {
            'rows': 2,
            'columns': 2,
            'spacing': 0.1
        }
    
    def _get_next_position(self) -> Dict[str, int]:
        """Get next available position in dashboard grid."""
        if self.current_dashboard is None:
            return {'row': 0, 'column': 0}
        
        dashboard = self.dashboards[self.current_dashboard]
        layout = dashboard['layout']
        plots = dashboard['plots']
        
        # Find occupied positions
        occupied = {
            (plot['position']['row'], plot['position']['column'])
            for plot in plots.values()
            if 'position' in plot
        }
        
        # Find first available position
        for row in range(layout['rows']):
            for col in range(layout['columns']):
                if (row, col) not in occupied:
                    return {'row': row, 'column': col}
        
        # Default to first position if none available
        return {'row': 0, 'column': 0}
    
    def _get_next_filter_position(self) -> Dict[str, int]:
        """Get next available filter position."""
        if self.current_dashboard is None:
            return {'order': 0}
            
        dashboard = self.dashboards[self.current_dashboard]
        return {'order': len(dashboard['filters'])}
    
    def _record_dashboard_action(
        self,
        action_type: str,
        target: str
    ) -> None:
        """Record dashboard action in history."""
        record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'action': action_type,
            'target': target,
            'dashboard': self.current_dashboard
        }
        
        self.dashboard_history.append(record)
        state_manager.set_state(
            f'visualization.dashboard_history.{len(self.dashboard_history)}',
            record
        )
    
    @monitor_performance
    def save_dashboard(
        self,
        dashboard_name: Optional[str] = None,
        path: Optional[Path] = None
    ) -> None:
        """Save dashboard configuration."""
        name = dashboard_name or self.current_dashboard
        if name is None:
            raise VisualizationError("No dashboard specified")
        
        if path is None:
            path = config.directories.base_dir / 'dashboards'
            path.mkdir(parents=True, exist_ok=True)
        
        dashboard = self.dashboards[name]
        
        # Save dashboard configuration
        config_path = path / f"{name}_config.json"
        dashboard_config = {
            'title': dashboard['title'],
            'layout': dashboard['layout'],
            'filters': dashboard['filters'],
            'metadata': dashboard['metadata']
        }
        
        with open(config_path, 'w') as f:
            json.dump(dashboard_config, f, indent=4)
        
        # Save plots separately
        plots_dir = path / name / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_name, plot_info in dashboard['plots'].items():
            plot_path = plots_dir / f"{plot_name}.html"
            plot_info['figure'].write_html(plot_path)
        
        logger.info(f"Dashboard saved to {path}")
    
    @monitor_performance
    def load_dashboard(
        self,
        path: Path
    ) -> None:
        """Load dashboard configuration."""
        if not path.exists():
            raise VisualizationError(f"Dashboard path not found: {path}")
        
        config_path = path / "config.json"
        if not config_path.exists():
            raise VisualizationError(f"Dashboard config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            dashboard_config = json.load(f)
        
        # Create dashboard
        name = path.stem
        self.create_dashboard(
            name,
            dashboard_config['title'],
            dashboard_config['layout']
        )
        
        # Load plots
        plots_dir = path / 'plots'
        if plots_dir.exists():
            for plot_path in plots_dir.glob('*.html'):
                plot_name = plot_path.stem
                fig = go.Figure(px.load_figure(plot_path))
                
                self.dashboards[name]['plots'][plot_name] = {
                    'figure': fig,
                    'position': dashboard_config['plots'][plot_name]['position'],
                    'type': dashboard_config['plots'][plot_name]['type'],
                    'kwargs': dashboard_config['plots'][plot_name]['kwargs']
                }
        
        # Load filters
        self.dashboards[name]['filters'] = dashboard_config['filters']
        self.dashboards[name]['metadata'] = dashboard_config['metadata']
        
        logger.info(f"Dashboard loaded from {path}")

# Create global dashboard manager instance
dashboard_manager = DashboardManager()