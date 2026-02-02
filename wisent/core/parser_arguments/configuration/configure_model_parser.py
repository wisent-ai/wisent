"""Parser setup for the 'configure-model' command."""


def setup_configure_model_parser(parser):
    """Set up the configure-model subcommand parser."""
    parser.add_argument("model", type=str, help="Model name to configure")
    parser.add_argument("--force", action="store_true", help="Force reconfiguration even if model already has a config")
