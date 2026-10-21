"""Switch the SocialShield colour theme.

Usage:  python src/set_theme.py midnight | light | terminal | slate
Then restart the app (close the window and run run_app.bat again).
"""
import sys

from themes import ROOT, THEMES, config_toml

if len(sys.argv) != 2 or sys.argv[1] not in THEMES:
    print("Choose one of:", ", ".join(THEMES))
    sys.exit(1)
name = sys.argv[1]
(ROOT / ".streamlit").mkdir(exist_ok=True)
(ROOT / ".streamlit/config.toml").write_text(config_toml(name), encoding="utf-8")
(ROOT / ".streamlit/theme.txt").write_text(name, encoding="utf-8")
print(f"Theme set to {THEMES[name]['label']}. Restart the app to see it.")
