"""Colour themes for SocialShield.

Change theme:  python src/set_theme.py <name>     (names: midnight, light, terminal, slate)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Alert colours: validated categorical palette, separate steps for light and dark backgrounds.
ALERTS_DARK = {"Phishing": "#3987e5", "Money Scam": "#d95926", "Spam": "#199e70", "Misinformation": "#c98500",
               "Impersonation": "#d55181", "Adult Scam": "#9085e9", "Safe": "#3f8f4f"}
ALERTS_LIGHT = {"Phishing": "#2a78d6", "Money Scam": "#eb6834", "Spam": "#1baf7a", "Misinformation": "#eda100",
                "Impersonation": "#e87ba4", "Adult Scam": "#6250d6", "Safe": "#008300"}

THEMES = {
    "midnight": {"label": "Midnight Blue", "base": "dark", "primary": "#3987e5", "bg": "#0e1117", "bg2": "#171b24",
                 "card": "#151922", "border": "#262b37", "text": "#f0efec", "muted": "#a3a7b3", "grid": "#262b37",
                 "neutral_bar": "#4a5a78", "node": "#8aa4c8", "edge": "#3a4050"},
    "light": {"label": "Clean Light", "base": "light", "primary": "#2a78d6", "bg": "#fcfcfb", "bg2": "#f1f0ec",
              "card": "#ffffff", "border": "#e2e1dc", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
              "neutral_bar": "#b4b9c4", "node": "#7a95bd", "edge": "#c9ccd3"},
    "terminal": {"label": "Cyber Terminal", "base": "dark", "primary": "#1fd17a", "bg": "#050807", "bg2": "#0c1410",
                 "card": "#09110d", "border": "#18291f", "text": "#d8f5e4", "muted": "#7fa391", "grid": "#16241d",
                 "neutral_bar": "#2f5a44", "node": "#5fbf8f", "edge": "#1f3a2c"},
    "slate": {"label": "Slate Teal", "base": "dark", "primary": "#2bb5a8", "bg": "#15181d", "bg2": "#1e232a",
              "card": "#1b2026", "border": "#2d343d", "text": "#eceef1", "muted": "#9aa3ad", "grid": "#2b313a",
              "neutral_bar": "#4b5a66", "node": "#8fb3c0", "edge": "#36404a"},
}
DEFAULT = "midnight"


def active_name() -> str:
    f = ROOT / ".streamlit/theme.txt"
    name = f.read_text(encoding="utf-8").strip() if f.exists() else DEFAULT
    return name if name in THEMES else DEFAULT


def active() -> dict:
    t = dict(THEMES[active_name()])
    t["alerts"] = ALERTS_LIGHT if t["base"] == "light" else ALERTS_DARK
    return t


def config_toml(name: str) -> str:
    t = THEMES[name]
    return (
        "[theme]\n"
        f'base = "{t["base"]}"\n'
        f'primaryColor = "{t["primary"]}"\n'
        f'backgroundColor = "{t["bg"]}"\n'
        f'secondaryBackgroundColor = "{t["bg2"]}"\n'
        f'textColor = "{t["text"]}"\n'
        "baseFontSize = 17\n\n"
        '[client]\ntoolbarMode = "minimal"\n\n'
        "[browser]\ngatherUsageStats = false\n"
    )
