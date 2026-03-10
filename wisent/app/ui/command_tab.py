"""Dynamic form builder for individual CLI commands.

Introspects the argparse parser setup function for each command to
generate Gradio UI components, Run button, and output area.
"""

import argparse
import importlib
import traceback
import gradio as gr
from wisent.core.utils.config_tools.constants import INDEX_FIRST, GRADIO_GALLERY_COLUMNS

_GALLERY_COLUMNS = GRADIO_GALLERY_COLUMNS
from wisent.app.ui.form_components import action_to_component, components_to_args
from wisent.app.core.runner import run_command


def build_command_tab(cmd_info):
    """Build a Gradio tab for a single CLI command.

    Args:
        cmd_info: CommandInfo with name, help_text, parser_module, parser_func.
    """
    gr.Markdown(f"**{cmd_info.name}** -- {cmd_info.help_text}")

    parser = _get_parser(cmd_info)
    if parser is None:
        gr.Markdown("*Could not load parser for this command.*")
        return

    components = []
    dests = []

    for action in parser._actions:
        comp, dest = action_to_component(action)
        if comp is not None:
            components.append(comp)
            dests.append((dest, action))

    run_btn = gr.Button(f"Run {cmd_info.name}", variant="primary")
    output = gr.Textbox(
        label="Output", interactive=False,
        elem_classes=["output-box"],
    )
    gallery = gr.Gallery(
        label="Visualizations", visible=True, columns=_GALLERY_COLUMNS,
    )

    if components:
        run_btn.click(
            fn=_make_handler(cmd_info.name, dests),
            inputs=components,
            outputs=[output, gallery],
        )
    else:
        run_btn.click(
            fn=lambda name=cmd_info.name: run_command(name, []),
            inputs=[],
            outputs=[output, gallery],
        )


def build_subparser_tab(cmd_info):
    """Build a tab for commands with sub-subparsers (e.g. optimize-steering).

    Shows a Dropdown for the sub-action, then renders all args for each
    sub-action in a shared form area.
    """
    gr.Markdown(f"**{cmd_info.name}** -- {cmd_info.help_text}")

    parser = _get_parser(cmd_info)
    if parser is None:
        gr.Markdown("*Could not load parser for this command.*")
        return

    sub_actions = _extract_subparsers(parser)
    if not sub_actions:
        build_command_tab(cmd_info)
        return

    sub_names = list(sub_actions.keys())
    first_sub = sub_names[INDEX_FIRST] if sub_names else None
    sub_dropdown = gr.Dropdown(
        label="Sub-action", choices=sub_names, value=first_sub,
    )

    all_components = []
    all_dests = []

    merged_actions = _merge_sub_actions(sub_actions)
    for action in merged_actions:
        comp, dest = action_to_component(action)
        if comp is not None:
            all_components.append(comp)
            all_dests.append((dest, action))

    run_btn = gr.Button(f"Run {cmd_info.name}", variant="primary")
    output = gr.Textbox(
        label="Output", interactive=False,
        elem_classes=["output-box"],
    )
    gallery = gr.Gallery(
        label="Visualizations", visible=True, columns=_GALLERY_COLUMNS,
    )

    inputs = [sub_dropdown] + all_components

    run_btn.click(
        fn=_make_subparser_handler(cmd_info.name, all_dests),
        inputs=inputs,
        outputs=[output, gallery],
    )


def _get_parser(cmd_info):
    """Import and invoke the parser setup function, return configured parser."""
    try:
        mod = importlib.import_module(cmd_info.parser_module)
        setup_fn = getattr(mod, cmd_info.parser_func)
        temp_parser = argparse.ArgumentParser(prog=cmd_info.name)
        setup_fn(temp_parser)
        return temp_parser
    except Exception:
        return None


def _extract_subparsers(parser):
    """Extract sub-subparsers dict from a parser (if any)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _merge_sub_actions(sub_parsers_dict):
    """Collect unique actions across all sub-subparsers by dest."""
    seen = set()
    merged = []
    for sub_parser in sub_parsers_dict.values():
        for action in sub_parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if action.dest not in seen:
                seen.add(action.dest)
                merged.append(action)
    return merged


def _make_handler(command_name, dests):
    """Create a click handler closure for a standard command."""
    def handler(*values):
        try:
            arg_list = components_to_args(values, dests, command_name)
            text, images = run_command(command_name, arg_list)
            return text, images or None
        except Exception:
            return traceback.format_exc(), None
    return handler


def _make_subparser_handler(command_name, dests):
    """Create a click handler closure for sub-subparser commands."""
    def handler(sub_action, *values):
        try:
            arg_list = [sub_action] if sub_action else []
            arg_list.extend(components_to_args(values, dests, command_name))
            text, images = run_command(command_name, arg_list)
            return text, images or None
        except Exception:
            return traceback.format_exc(), None
    return handler
