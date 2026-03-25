"""Build the complete tabbed Gradio interface for Wisent.

Creates a model selector header, a live resource monitor bar, and
one tab per command group, each containing sub-tabs for individual
commands.
"""

import base64
import os
import platform

import gradio as gr
import psutil

from wisent.core.utils.config_tools.constants import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    INDEX_FIRST,
    GRADIO_MODEL_EXAMPLES,
    GRADIO_RESOURCE_BAR_HEIGHT_PX,
    GRADIO_RESOURCE_BAR_MARGIN_PX,
    GRADIO_RESOURCE_BAR_RADIUS_PX,
    GRADIO_RESOURCE_BAR_WIDTH_PX,
    GRADIO_RESOURCE_FONT_SIZE_PX,
    GRADIO_RESOURCE_GPU_COUNT_SINGLE,
    GRADIO_RESOURCE_LABEL_MARGIN_PX,
    GRADIO_RESOURCE_PADDING_H_PX,
    GRADIO_RESOURCE_PADDING_V_PX,
    GRADIO_RESOURCE_REFRESH_SECONDS,
    GRADIO_RESOURCE_SECTION_MARGIN_PX,
    PERCENT_MULTIPLIER,
    WISENT_COLOR_CHARCOAL,
    WISENT_COLOR_DARK_SURFACE,
    WISENT_COLOR_MINT,
    WISENT_COLOR_MINT_ACCENT_DARK,
    WISENT_COLOR_LIGHT_TEXT_MUTED,
    WISENT_COLOR_TEXT_LIGHT,
    WISENT_COLOR_TEXT_MUTED,
    WISENT_LOGO_DISPLAY_WIDTH,
    WISENT_LOGO_FILENAME,
)
from wisent.app.core.groups import get_command_groups
from wisent.app.ui.command_tab import build_command_tab, build_subparser_tab
from wisent.app.ui.wiring.navigation import wire_wizard_navigation
from wisent.app.ui.wizard import build_wizard_tab
from wisent.app.ui.tabs.benchmark_debug import build_benchmark_debug_tab

_SUBPARSER_COMMANDS = frozenset({"optimize-steering", "inference-config"})


def _find_logo():
    """Locate the logo file relative to the app package."""
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(app_dir, WISENT_LOGO_FILENAME)
    if os.path.exists(path):
        return path
    return None


# --- Resource monitor helpers ---


def _is_zerogpu():
    """Return True when running on a HuggingFace ZeroGPU Space."""
    try:
        import spaces
        return hasattr(spaces, "GPU")
    except ImportError:
        return False


def _get_gpu_info():
    """Return (gpu_name, gpu_count, gpu_mem_used_mb, gpu_mem_total_mb)."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(INDEX_FIRST)
            mem_used = (
                torch.cuda.memory_allocated(INDEX_FIRST) // BYTES_PER_MB
            )
            props = torch.cuda.get_device_properties(INDEX_FIRST)
            mem_total = props.total_memory // BYTES_PER_MB
            return name, count, mem_used, mem_total
    except Exception:
        pass
    if _is_zerogpu():
        return "H200 (ZeroGPU)", GRADIO_RESOURCE_GPU_COUNT_SINGLE, None, None
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)", GRADIO_RESOURCE_GPU_COUNT_SINGLE, None, None
    except Exception:
        pass
    return None, INDEX_FIRST, None, None


def _bar_css():
    """Return the progress bar container CSS."""
    return (
        f"display:inline-block;"
        f"width:{GRADIO_RESOURCE_BAR_WIDTH_PX}px;"
        f"height:{GRADIO_RESOURCE_BAR_HEIGHT_PX}px;"
        f"background:{WISENT_COLOR_LIGHT_TEXT_MUTED};"
        f"border-radius:{GRADIO_RESOURCE_BAR_RADIUS_PX}px;"
        f"overflow:hidden;vertical-align:middle;"
        f"margin:{INDEX_FIRST} {GRADIO_RESOURCE_BAR_MARGIN_PX}px;"
    )


def _fill_css(clamped):
    """Return the progress bar fill CSS."""
    return (
        f"width:{clamped}%;height:{PERCENT_MULTIPLIER}%;"
        f"background:{WISENT_COLOR_MINT};"
        f"border-radius:{GRADIO_RESOURCE_BAR_RADIUS_PX}px;"
    )


def _section_style():
    """Return per-section margin and text color CSS."""
    return (
        f"margin-right:{GRADIO_RESOURCE_SECTION_MARGIN_PX}px;"
        f"color:{WISENT_COLOR_TEXT_LIGHT};"
    )


def _build_progress_bar(percent):
    """Build an inline CSS progress bar."""
    clamped = max(INDEX_FIRST, min(PERCENT_MULTIPLIER, percent))
    return (
        f'<div style="{_bar_css()}">'
        f'<div style="{_fill_css(clamped)}"></div>'
        f'</div>'
    )


def _label_span(text):
    """Build a mint-colored label span."""
    return (
        f'<span style="color:{WISENT_COLOR_MINT};'
        f'font-weight:bold;">{text}</span> '
    )


def _gpu_section():
    """Build the GPU section of the resource bar."""
    gpu_name, gpu_count, gpu_mem_used, gpu_mem_total = _get_gpu_info()
    if gpu_name:
        gpu_label = (
            f"{gpu_count} &times; {gpu_name}"
            if gpu_count > GRADIO_RESOURCE_GPU_COUNT_SINGLE
            else gpu_name
        )
        if gpu_mem_total is not None and gpu_mem_total > INDEX_FIRST:
            gpu_pct = gpu_mem_used / gpu_mem_total * PERCENT_MULTIPLIER
            return (
                f'<span style="{_section_style()}">'
                f'{_label_span("GPU")}{gpu_label} &mdash; '
                f'{gpu_mem_used} / {gpu_mem_total} MB'
                f'{_build_progress_bar(gpu_pct)}</span>'
            )
        return (
            f'<span style="{_section_style()}">'
            f'{_label_span("GPU")}{gpu_label}</span>'
        )
    return (
        f'<span style="{_section_style()}">'
        f'<span style="color:{WISENT_COLOR_TEXT_MUTED};">'
        f'GPU</span> None</span>'
    )


def _container_css():
    """Return the outer container CSS."""
    container_radius = GRADIO_RESOURCE_BAR_RADIUS_PX + GRADIO_RESOURCE_BAR_RADIUS_PX
    return (
        f"display:flex;align-items:center;flex-wrap:wrap;"
        f"padding:{GRADIO_RESOURCE_PADDING_V_PX}px "
        f"{GRADIO_RESOURCE_PADDING_H_PX}px;"
        f"background:{WISENT_COLOR_CHARCOAL};"
        f"border-radius:{container_radius}px;"
        f"font-size:{GRADIO_RESOURCE_FONT_SIZE_PX}px;"
        f"color:{WISENT_COLOR_TEXT_LIGHT};"
        f"font-family:monospace;"
    )


def _format_resource_html():
    """Generate the HTML for the resource monitor bar."""
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / BYTES_PER_GB
    ram_total_gb = ram.total / BYTES_PER_GB
    ram_percent = ram.percent

    cpu_html = (
        f'<span style="{_section_style()}">'
        f'{_label_span("CPU")}{cpu_percent:.0f}%'
        f'{_build_progress_bar(cpu_percent)}</span>'
    )
    ram_html = (
        f'<span style="{_section_style()}">'
        f'{_label_span("RAM")}{ram_used_gb:.1f} / {ram_total_gb:.0f} GB'
        f'{_build_progress_bar(ram_percent)}</span>'
    )

    system_label = platform.node() or "local"
    label_css = (
        f"color:{WISENT_COLOR_MINT};"
        f"font-weight:bold;"
        f"margin-right:{GRADIO_RESOURCE_LABEL_MARGIN_PX}px;"
    )
    return (
        f'<div style="{_container_css()}">'
        f'<span style="{label_css}">{system_label}</span>'
        f'{_gpu_section()}{cpu_html}{ram_html}</div>'
    )


def _build_resource_monitor():
    """Build the live resource monitor component with auto-refresh."""
    timer = gr.Timer(value=GRADIO_RESOURCE_REFRESH_SECONDS)
    gr.HTML(value=_format_resource_html, every=timer)


# --- Main interface builder ---


def build_interface():
    """Construct the full Gradio interface inside an active gr.Blocks context."""
    logo_path = _find_logo()
    with gr.Row(equal_height=False):
        if logo_path:
            with open(logo_path, "rb") as _f:
                _b64 = base64.b64encode(_f.read()).decode()
            with gr.Column(
                scale=INDEX_FIRST, min_width=WISENT_LOGO_DISPLAY_WIDTH,
            ):
                gr.HTML(
                    f'<img src="data:image/png;base64,{_b64}" '
                    f'width="{WISENT_LOGO_DISPLAY_WIDTH}" '
                    f'style="display:block;" />'
                )
        with gr.Column():
            gr.Markdown(
                "# Wisent\n"
                "### World's Best AI through Representation Engineering\n"
                "Select a category tab, choose a command, "
                "fill in parameters, and click Run."
            )

    with gr.Row():
        gr.Dropdown(
            label="Model", choices=list(GRADIO_MODEL_EXAMPLES),
            value=None, allow_custom_value=True, interactive=True,
            elem_id="global-model")
        _build_resource_monitor()
    gr.HTML('<script>new MutationObserver(function(m,o){var e=document.querySelector("#global-model input[type=text]");if(e&&!e.placeholder){e.placeholder="Type your model HuggingFace ID or select from dropdown";o.disconnect();console.log("WISENT: placeholder set")}}).observe(document.body,{childList:true,subtree:true});console.log("WISENT: observer started")</script>')

    groups = get_command_groups()

    outer_tabs = gr.Tabs(elem_id="main-tabs")
    inner_tabs_map = {}
    with outer_tabs:
        with gr.Tab(label="Wizard", id="wizard"):
            wizard_components = build_wizard_tab()
        with gr.Tab(label="Benchmark Debug", id="benchmark-debug"):
            build_benchmark_debug_tab()
        for group in groups:
            with gr.Tab(label=group.label, id=group.label):
                gr.Markdown(f"*{group.description}*")
                group_tabs = gr.Tabs(elem_id=f"tabs-{group.label}")
                inner_tabs_map[group.label] = group_tabs
                with group_tabs:
                    for cmd in group.commands:
                        with gr.Tab(label=cmd.name, id=cmd.name):
                            if cmd.name in _SUBPARSER_COMMANDS:
                                build_subparser_tab(cmd)
                            else:
                                build_command_tab(cmd)

    wire_wizard_navigation(
        groups, wizard_components, outer_tabs, inner_tabs_map,
    )
