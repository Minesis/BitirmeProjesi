"""
run_ui.py
---------
Single entrypoint to launch the Streamlit UI without using `streamlit run ...`.

Usage (Windows):
  .\\venv\\Scripts\\python.exe run_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).parent
    ui_script = project_root / "app" / "ui" / "control_panel.py"

    if not ui_script.exists():
        raise FileNotFoundError(f"UI script not found: {ui_script}")

    # Streamlit exposes an internal CLI entrypoint we can call directly.
    # This avoids requiring users to type `streamlit run ...`.
    sys.argv = [
        "streamlit",
        "run",
        str(ui_script),
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]

    try:
        from streamlit.web.cli import main as st_main  # type: ignore
    except Exception:
        # Fallback for older/newer streamlit layouts.
        from streamlit.web import cli as st_cli  # type: ignore
        st_main = st_cli.main  # type: ignore[attr-defined]

    st_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
