from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import colorsys

from core.config import config
from core.exceptions import VisualizationError
from core.state_manager import state_manager
from utils.loggers import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

@dataclass
class Theme:
    """Theme configuration for visualizations."""
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    background_color: str = '#ffffff'
    text_color: str = '#2f2f2f'
    grid_color: str = '#e6e6e6'
    font_family: str = 'Arial, sans-serif'
    font_size: int = 12
    title_font_size: int = 16
    axis_font_size: int = 10
    legend_font_size: int = 10
    color_sequence: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ])
    template: str = 'plotly_white'

class StyleManager:
    """Manage visualization styles and themes."""
    
    def __init__(self):
        self.current_theme = Theme()
        self.themes: Dict[str, Theme] = {
            'default': Theme(),
            'dark': self._create_dark_theme(),
            'light': self._create_light_theme()
        }
        self.style_history: List[Dict[str, Any]] = []
    
    def _create_dark_theme(self) -> Theme:
        """Create dark theme."""
        return Theme(
            primary_color='#61dafb',
            secondary_color='#fb8761',
            background_color='#2f2f2f',
            text_color='#ffffff',
            grid_color='#404040',
            color_sequence=[
                '#61dafb', '#fb8761', '#2ecc71', '#e74c3c',
                '#9b59b6', '#e67e22', '#f1c40f', '#bdc3c7'
            ],
            template='plotly_dark'
        )
    
    def _create_light_theme(self) -> Theme:
        """Create light theme."""
        return Theme(
            primary_color='#1f77b4',
            secondary_color='#ff7f0e',
            background_color='#ffffff',
            text_color='#2f2f2f',
            grid_color='#e6e6e6',
            color_sequence=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
            ],
            template='plotly_white'
        )
    
    @monitor_performance
    def set_theme(self, theme_name: str) -> None:
        """Set current theme."""
        if theme_name not in self.themes:
            raise VisualizationError(f"Theme not found: {theme_name}")
        
        self.current_theme = self.themes[theme_name]
        state_manager.set_state('visualization.current_theme', theme_name)
        self._record_style_change('theme_change', {'theme': theme_name})
    
    @monitor_performance
    def create_custom_theme(
        self,
        name: str,
        **theme_params
    ) -> None:
        """Create custom theme."""
        theme = Theme(**theme_params)
        self.themes[name] = theme
        self._record_style_change('theme_creation', {'name': name})
    
    @monitor_performance
    def apply_theme_to_figure(
        self,
        fig: go.Figure,
        override_params: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Apply current theme to figure."""
        theme = self.current_theme
        
        # Apply overrides if provided
        if override_params:
            for key, value in override_params.items():
                if hasattr(theme, key):
                    setattr(theme, key, value)
        
        # Update layout with theme settings
        fig.update_layout(
            template=theme.template,
            paper_bgcolor=theme.background_color,
            plot_bgcolor=theme.background_color,
            font=dict(
                family=theme.font_family,
                size=theme.font_size,
                color=theme.text_color
            ),
            title=dict(
                font=dict(
                    size=theme.title_font_size,
                    color=theme.text_color
                )
            ),
            xaxis=dict(
                gridcolor=theme.grid_color,
                tickfont=dict(
                    size=theme.axis_font_size,
                    color=theme.text_color
                )
            ),
            yaxis=dict(
                gridcolor=theme.grid_color,
                tickfont=dict(
                    size=theme.axis_font_size,
                    color=theme.text_color
                )
            ),
            legend=dict(
                font=dict(
                    size=theme.legend_font_size,
                    color=theme.text_color
                )
            )
        )
        
        # Update color sequence
        for trace in fig.data:
            if hasattr(trace, 'marker'):
                trace.marker.color = theme.color_sequence[
                    fig.data.index(trace) % len(theme.color_sequence)
                ]
        
        return fig
    
    @monitor_performance
    def generate_color_palette(
        self,
        n_colors: int,
        base_color: Optional[str] = None
    ) -> List[str]:
        """Generate color palette."""
        if base_color is None:
            base_color = self.current_theme.primary_color
        
        # Convert hex to HSV
        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        hsv = colorsys.rgb_to_hsv(*rgb)
        
        # Generate colors with varying hue
        colors = []
        for i in range(n_colors):
            hue = (hsv[0] + i/n_colors) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, hsv[1], hsv[2])
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    @monitor_performance
    def style_grid(
        self,
        fig: go.Figure,
        show_grid: bool = True,
        grid_width: Optional[float] = None,
        grid_style: Optional[str] = None
    ) -> go.Figure:
        """Style figure grid."""
        grid_update = dict(
            showgrid=show_grid,
            gridcolor=self.current_theme.grid_color
        )
        
        if grid_width is not None:
            grid_update['gridwidth'] = grid_width
        if grid_style is not None:
            grid_update['gridstyle'] = grid_style
        
        fig.update_xaxes(**grid_update)
        fig.update_yaxes(**grid_update)
        
        return fig
    
    @monitor_performance
    def style_legend(
        self,
        fig: go.Figure,
        position: Optional[str] = None,
        orientation: Optional[str] = None
    ) -> go.Figure:
        """Style figure legend."""
        legend_update = dict(
            font=dict(
                family=self.current_theme.font_family,
                size=self.current_theme.legend_font_size,
                color=self.current_theme.text_color
            )
        )
        
        if position is not None:
            x, y = {
                'top-right': (0.95, 0.95),
                'top-left': (0.05, 0.95),
                'bottom-right': (0.95, 0.05),
                'bottom-left': (0.05, 0.05)
            }.get(position, (0.95, 0.95))
            legend_update.update(x=x, y=y)
        
        if orientation is not None:
            legend_update['orientation'] = orientation
        
        fig.update_layout(legend=legend_update)
        return fig
    
    def _record_style_change(
        self,
        change_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Record style change in history."""
        record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'type': change_type,
            'details': details
        }
        
        self.style_history.append(record)
        state_manager.set_state(
            f'visualization.style_history.{len(self.style_history)}',
            record
        )
    
    @monitor_performance
    def save_theme(
        self,
        theme_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Save theme to file."""
        if theme_name not in self.themes:
            raise VisualizationError(f"Theme not found: {theme_name}")
        
        if path is None:
            path = config.directories.base_dir / 'themes'
            path.mkdir(parents=True, exist_ok=True)
        
        theme = self.themes[theme_name]
        theme_dict = {
            key: value for key, value in theme.__dict__.items()
        }
        
        theme_path = path / f"{theme_name}.json"
        with open(theme_path, 'w') as f:
            json.dump(theme_dict, f, indent=4)
        
        logger.info(f"Theme saved to {theme_path}")
    
    @monitor_performance
    def load_theme(
        self,
        path: Path
    ) -> None:
        """Load theme from file."""
        if not path.exists():
            raise VisualizationError(f"Theme file not found: {path}")
        
        with open(path, 'r') as f:
            theme_dict = json.load(f)
        
        theme_name = path.stem
        self.themes[theme_name] = Theme(**theme_dict)
        logger.info(f"Theme loaded from {path}")

# Create global style manager instance
style_manager = StyleManager()