"""Wire wizard command selection to tab navigation."""

import gradio as gr
from wisent.core.utils.config_tools.constants import COMBO_OFFSET


def wire_wizard_navigation(groups, wizard_components, outer_tabs, inner):
    """Connect the wizard 'Go' button to navigate to the command tab.

    Args:
        groups: List of CommandGroup objects.
        wizard_components: Tuple of (go_button, cmd_state) from wizard.
        outer_tabs: The main gr.Tabs component.
        inner: Dict mapping group label to its inner gr.Tabs component.
    """
    go_btn, cmd_state = wizard_components
    cmd_to_group = {}
    for group in groups:
        for cmd in group.commands:
            cmd_to_group[cmd.name] = group.label

    inner_list = list(inner.items())

    def navigate(cmd_name=None):
        if not cmd_name:
            return [gr.update()] * (len(inner_list) + COMBO_OFFSET)
        group_label = cmd_to_group.get(cmd_name)
        if not group_label:
            return [gr.update()] * (len(inner_list) + COMBO_OFFSET)
        outputs = [gr.Tabs(selected=group_label)]
        for label, _tabs in inner_list:
            if label == group_label:
                outputs.append(gr.Tabs(selected=cmd_name))
            else:
                outputs.append(gr.update())
        return outputs

    all_outputs = [outer_tabs] + [t for _, t in inner_list]
    go_btn.click(fn=navigate, inputs=[cmd_state], outputs=all_outputs)
