"""
Wisent project branding assets (ASCII art logo, project description).

This module provides reusable branding components that can be used by:
- CLI interfaces
- Documentation
- Web interfaces
- Any other presentation layer

The ASCII art represents a Wisent (European bison), the project's namesake.
"""

from __future__ import annotations

__all__ = ["WISENT_ASCII_LOGO", "PROJECT_TAGLINE", "render_banner", "get_logo"]


WISENT_ASCII_LOGO = """
.................  .:--++*##%%%%##**+=-:.  .................
..             .:=*%@@@@@@@%%%%%%%@@@@@@%*=:.             ..
.           .-*%@@@%#+=-::.........:-=+#%@@@%*=.           .
.         -*%@@@#=:.                    .:=*%@@@*-.        .
.      .-#@@@*=.                            .-*@@@#-.      .
.     :#@@@*:                                  :+%@@#-     .
.   .+@@@*:                                      :+@@@+.   .
.  .*@@@@%*=:.                                     -%@@#:  .
. .#@@#=*%@@@%*-:.                                  .#@@%: .
..*@@%.  .-+#@@@@#+-:.                               .*@@%..
.=@@@-       :-+#@@@@%*=:.                            .%@@*.
:#@@+           .:-+#@@@@%#+=:.                        -@@@-
=@@@:                .-=*%@@@@%#+=:..                  .#@@+
+@@@*=:.                 .:-+*%@@@@%#*=-:..             *@@+
+@@@@@@#+-..                  .:-=*#@@@@@%#*+--..       +@@+
+@@#-+%@@@%:                        .:-=*#%@@@@@%#*+=-:.*@@+
=@@%. .=@@@:                             ..:-=+#%%@@@@@%@@@+
:%@@=  :@@@-                                    ..::-=+#@@@=
.+@@%. .#@@*                                           +@@#:
..%@@*. =@@@:                                         =@@@-.
. :%@@*..#@@#.                         .:..          =@@@= .
.  :%@@*.:%@@*.                       :#@@%#*+=-::..+@@@=  .
.   :#@@%-:%@@#:                    .+@@@#%%@@@@@@%%@@%-   .
.    .+@@@*=#@@%-                 .=%@@%=...::-=#@@@@*.    .
.      :*@@@#%@@@*:             .=%@@@+.     .:*%@@#-      .
.        :+%@@@@@@@*-.       :=*@@@%+.    .-+%@@@*-.       .
.          .=*%@@@@@@#+:.:-+#@@@%*-. .:-+#%@@@#+:          .
.             .-+#%@@@@@@@@@@@@#*+**#@@@@@%*=:.            .
..............   ..-=+*#%%%@@@@@@@@%%#*=-:.   ..............
 ...................  ....:::::::::.... ...................
""".strip()

PROJECT_TAGLINE = "Steering vectors & activation tooling"


def get_logo(width: int = 48) -> str:
    """
    Get the Wisent ASCII logo, optionally centered to a specific width.

    Args:
        width: Total width to center the logo within (default: 48)

    Returns:
        ASCII art logo as a string
    """
    return "\n".join(line.center(width) for line in WISENT_ASCII_LOGO.splitlines())


def render_banner(title: str, width: int = 48, use_color: bool = True) -> str:
    """
    Render a banner with the Wisent logo, title, and tagline.

    Args:
        title: Title text to display (e.g., "Wisent CLI")
        width: Width for centering (default: 48)
        use_color: Whether to use ANSI color codes (default: True)

    Returns:
        Formatted banner as a string
    """
    logo = get_logo(width)

    if use_color:
        GREEN = "\x1b[32m"
        BOLD = "\x1b[1m"
        OFF = "\x1b[0m"
        banner = f"{GREEN}{logo}{OFF}\n"
        banner += f"{BOLD}{GREEN}{title}{OFF} – {PROJECT_TAGLINE}\n"
    else:
        banner = f"{logo}\n"
        banner += f"{title} – {PROJECT_TAGLINE}\n"

    return banner


def print_banner(title: str, width: int = 48, use_color: bool = True) -> None:
    """
    Print a banner with the Wisent logo, title, and tagline.

    Args:
        title: Title text to display (e.g., "Wisent CLI")
        width: Width for centering (default: 48)
        use_color: Whether to use ANSI color codes (default: True)
    """
    print(render_banner(title, width, use_color))


if __name__ == "__main__":
    # Demo the branding
    print_banner("Wisent")
