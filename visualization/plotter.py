import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from core.config import config
from core.exceptions import VisualizationError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class Plotter:
    """Handle all visualization operations."""
    
    def __init__(self):
        self.plot_history: List[Dict[str, Any]] = []
        self.figures: Dict[str, go.Figure] = {}
        self.default_height = config.ui.chart_height
        self.default_width = config.ui.chart_width
        self.theme = config.ui.streamlit_theme
    
    @monitor_performance
    @handle_exceptions(VisualizationError)
    def create_plot(
        self,
        plot_type: str,
        data: Any,
        **kwargs
    ) -> go.Figure:
        """Create plot based on type."""
        plot_func = getattr(self, f'_create_{plot_type}_plot', None)
        if plot_func is None:
            raise VisualizationError(
                f"Unsupported plot type: {plot_type}"
            )
        
        # Add default styling
        kwargs.setdefault('height', self.default_height)
        kwargs.setdefault('width', self.default_width)
        
        fig = plot_func(data, **kwargs)
        
        # Store plot in history
        self._record_plot(plot_type, kwargs)
        
        return fig
    
    @monitor_performance
    def _create_scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create scatter plot."""
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            legend_title_text=color if color else None
        )
        
        return fig
    
    @monitor_performance
    def _create_line_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create line plot."""
        fig = px.line(
            data,
            x=x,
            y=y,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    @monitor_performance
    def _create_bar_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        orientation: str = 'v',
        **kwargs
    ) -> go.Figure:
        """Create bar plot."""
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            orientation=orientation,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            bargap=0.2
        )
        
        return fig
    
    @monitor_performance
    def _create_histogram_plot(
        self,
        data: pd.DataFrame,
        column: str,
        bins: Optional[int] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create histogram plot."""
        fig = px.histogram(
            data,
            x=column,
            nbins=bins,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            bargap=0.1
        )
        
        return fig
    
    @monitor_performance
    def _create_box_plot(
        self,
        data: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create box plot."""
        fig = px.box(
            data,
            x=by,
            y=column,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if by else False
        )
        
        return fig
    
    @monitor_performance
    def _create_violin_plot(
        self,
        data: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create violin plot."""
        fig = px.violin(
            data,
            x=by,
            y=column,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if by else False
        )
        
        return fig
    
    @monitor_performance
    def _create_heatmap_plot(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create heatmap plot."""
        fig = px.imshow(
            data,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig
    
    @monitor_performance
    def _create_pie_plot(
        self,
        data: pd.DataFrame,
        values: str,
        names: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create pie plot."""
        fig = px.pie(
            data,
            values=values,
            names=names,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    @monitor_performance
    def _create_density_plot(
        self,
        data: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create density plot."""
        fig = ff.create_distplot(
            [data[column].dropna()],
            [column],
            show_hist=False,
            show_rug=False
        )
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            template='plotly_white',
            showlegend=False,
            **kwargs
        )
        
        return fig
    
    @monitor_performance
    def _create_scatter_matrix(
        self,
        data: pd.DataFrame,
        columns: List[str],
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create scatter matrix plot."""
        fig = px.scatter_matrix(
            data,
            dimensions=columns,
            color=color,
            title=title,
            template='plotly_white',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if color else False
        )
        
        return fig
    
    @monitor_performance
    def _create_parallel_coordinates(
        self,
        data: pd.DataFrame,
        columns: List[str],
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create parallel coordinates plot."""
        fig = px.parallel_coordinates(
            data,
            dimensions=columns,
            color=color,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if color else False
        )
        
        return fig
    
    def _record_plot(
        self,
        plot_type: str,
        plot_args: Dict[str, Any]
    ) -> None:
        """Record plot creation in history."""
        record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'plot_type': plot_type,
            'arguments': plot_args
        }
        
        self.plot_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'visualization.history.{len(self.plot_history)}',
            record
        )
    
    @monitor_performance
    def save_plot(
        self,
        fig: go.Figure,
        path: Path,
        format: str = 'html',
        **kwargs
    ) -> None:
        """Save plot to file."""
        try:
            if format == 'html':
                fig.write_html(path, **kwargs)
            elif format == 'png':
                fig.write_image(path, **kwargs)
            elif format == 'json':
                fig.write_json(path, **kwargs)
            else:
                raise VisualizationError(f"Unsupported format: {format}")
                
            logger.info(f"Plot saved to {path}")
            
        except Exception as e:
            raise VisualizationError(
                f"Error saving plot: {str(e)}",
                details={'path': str(path), 'format': format}
            ) from e
    
    @monitor_performance
    def get_plot_summary(self) -> Dict[str, Any]:
        """Get summary of plot creation history."""
        return {
            'total_plots': len(self.plot_history),
            'plot_types': list(set(
                record['plot_type'] for record in self.plot_history
            )),
            'recent_plots': self.plot_history[-5:]
        }

# Create global plotter instance
plotter = Plotter()