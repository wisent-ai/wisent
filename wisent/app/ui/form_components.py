"""Map argparse actions to Gradio components and back.

Introspects argparse action types and creates matching Gradio form
elements. Also provides reverse mapping from Gradio values to CLI args.
"""

import argparse
import gradio as gr
from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    GRADIO_APPEND_LINES,
)


def action_to_component(action):
    """Convert an argparse action to a Gradio component.

    Returns (component, arg_name) or (None, None) for skip actions.
    """
    if isinstance(action, argparse._HelpAction):
        return None, None
    if isinstance(action, argparse._SubParsersAction):
        return None, None

    dest = action.dest
    if dest == "command":
        return None, None

    label = _build_label(action)
    default = action.default

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        val = bool(default) if default is not None else False
        return gr.Checkbox(label=label, value=val), dest

    if action.choices is not None:
        choices_list = [str(c) for c in action.choices]
        val = str(default) if default is not None else None
        return gr.Dropdown(
            label=label, choices=choices_list, value=val,
            allow_custom_value=True,
        ), dest

    if action.type is int:
        val = default if isinstance(default, int) else None
        return gr.Number(label=label, value=val, precision=INDEX_FIRST), dest

    if action.type is float:
        val = default if isinstance(default, (int, float)) else None
        return gr.Number(label=label, value=val), dest

    if isinstance(action, argparse._AppendAction):
        return gr.Textbox(
            label=label, value=str(default) if default else "",
            lines=GRADIO_APPEND_LINES, placeholder="One value per line",
        ), dest

    if action.nargs in ("+", "*"):
        val = " ".join(str(v) for v in default) if default else ""
        return gr.Textbox(
            label=label, value=val,
            placeholder="Space-separated values",
        ), dest

    val = str(default) if default is not None else ""
    required = action.required if hasattr(action, "required") else False
    is_positional = not action.option_strings
    placeholder = "Required" if (required or is_positional) else ""
    return gr.Textbox(label=label, value=val, placeholder=placeholder), dest


def _build_label(action):
    """Build a human-readable label from an argparse action."""
    if action.option_strings:
        name = action.option_strings[INDEX_FIRST]
        for opt in action.option_strings:
            if opt.startswith("--"):
                name = opt
                break
        name = name.lstrip("-").replace("-", " ").replace("_", " ").title()
    else:
        name = action.dest.replace("_", " ").replace("-", " ").title()

    if action.help and action.help != argparse.SUPPRESS:
        return f"{name} -- {action.help}"
    return name


def components_to_args(component_values, component_dests, command_name):
    """Convert Gradio component values back to CLI argument list.

    Args:
        component_values: List of values from Gradio components.
        component_dests: List of (dest_name, action) tuples.
        command_name: The CLI command name.

    Returns:
        List of CLI argument strings.
    """
    args = []
    for value, (dest, action) in zip(component_values, component_dests):
        if value is None or value == "":
            continue

        if isinstance(action, (argparse._StoreTrueAction,)):
            if value:
                flag = _dest_to_flag(dest)
                args.append(flag)
            continue

        if isinstance(action, argparse._StoreFalseAction):
            if not value:
                flag = _dest_to_flag(dest)
                args.append(flag)
            continue

        if not action.option_strings:
            args.append(str(value))
            continue

        flag = _dest_to_flag(dest)

        if isinstance(action, argparse._AppendAction):
            for line in str(value).strip().splitlines():
                line = line.strip()
                if line:
                    args.extend([flag, line])
            continue

        if action.nargs in ("+", "*"):
            parts = str(value).strip().split()
            if parts:
                args.append(flag)
                args.extend(parts)
            continue

        args.extend([flag, str(value)])
    return args


def _dest_to_flag(dest):
    """Convert argparse dest back to CLI flag string."""
    return "--" + dest.replace("_", "-")
