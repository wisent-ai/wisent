"""Patch visualizations module to use working PaCMAP implementation."""
from wisent.core.utils.visualization.geometry.public import visualizations
from .pacmap_alt import plot_pacmap_alt

# Replace the broken plot_pacmap_projection with working alternative
visualizations.plot_pacmap_projection = plot_pacmap_alt
