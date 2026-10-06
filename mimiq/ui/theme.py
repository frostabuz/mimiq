"""Mimiq design tokens and the global Qt stylesheet.

A calm, neutral "tool" look: graphite surfaces, one solid accent used only for the active state and the main
action, small radii, no gradients or glow.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QLinearGradient, QPalette

BG = "#111113"          # window
PANEL = "#18181B"       # side panels, control bar
CARD = "#1D1D20"        # list items, popups
CARD_HI = "#26262A"     # hover / raised
FIELD = "#222226"       # inputs, segmented track
SEG_ON = "#3A3A40"      # selected segment (neutral, like a native segmented control)
LINE = "#2A2A2E"
LINE_HI = "#38383E"
TEXT = "#ECECEE"
DIM = "#A0A0A7"
FAINT = "#6B6B73"
ACCENT = "#3D8BFD"
ACCENT_HI = "#5B9DFD"
ACCENT_LO = "#2F72D6"
ACCENT_SOFT = "rgba(61,139,253,0.16)"
OK, WARN, BAD = "#34C759", "#FF9F0A", "#FF453A"
LIVE = "#FF3B30"

# kept for older call sites: the accent is a single solid colour now
A1 = A2 = A3 = ACCENT
GRAD = ACCENT
GRAD_HOVER = ACCENT_HI


def accent_gradient(x1: float, y1: float, x2: float, y2: float) -> QLinearGradient:
    """Solid accent as a brush (compatibility helper — Mimiq no longer uses colour gradients)."""
    g = QLinearGradient(x1, y1, x2, y2)
    g.setColorAt(0.0, QColor(ACCENT))
    g.setColorAt(1.0, QColor(ACCENT))
    return g


def ui_font(size: float = 10.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    families = set(QFontDatabase.families())
    for fam in ("Segoe UI Variable Text", "Segoe UI", "Inter", "SF Pro Text", "Noto Sans", "Liberation Sans",
                "Arial"):
        if fam in families:
            break
    else:
        fam = QFont().family()
    f = QFont(fam)
    f.setPointSizeF(size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def palette() -> QPalette:
    p = QPalette()
    for role, color in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, FIELD),
                        (QPalette.ColorRole.AlternateBase, CARD), (QPalette.ColorRole.Text, TEXT),
                        (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Button, CARD),
                        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
                        (QPalette.ColorRole.HighlightedText, "#FFFFFF"), (QPalette.ColorRole.ToolTipBase, CARD_HI),
                        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, FAINT)):
        p.setColor(role, QColor(color))
    return p


STYLESHEET = f"""
* {{ outline: none; }}
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QMainWindow, #root {{ background: {BG}; }}
QToolTip {{ background: {CARD_HI}; color: {TEXT}; border: 1px solid {LINE_HI}; border-radius: 4px; padding: 5px 8px; }}

#header {{ background: {BG}; }}
#sidePanel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; }}
#card {{ background: transparent; border: none; border-top: 1px solid {LINE}; border-radius: 0; }}
#cardFlat {{ background: transparent; border: none; }}
#box {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; }}
#controlBar {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; }}
#statusBar {{ background: {BG}; }}

QLabel#h1 {{ font-size: 14pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#section {{ color: {DIM}; font-size: 9pt; font-weight: 600; }}
QLabel#dim {{ color: {DIM}; }}
QLabel#faint {{ color: {FAINT}; font-size: 9pt; }}
QLabel#value {{ color: {TEXT}; font-weight: 600; }}
QLabel#badge {{ color: {DIM}; background: transparent; border: 1px solid {LINE_HI}; border-radius: 4px;
                padding: 1px 6px; font-size: 8.5pt; }}

QPushButton {{ background: {FIELD}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 7px 14px;
               color: {TEXT}; font-weight: 500; }}
QPushButton:hover {{ background: {CARD_HI}; }}
QPushButton:pressed {{ background: {CARD}; }}
QPushButton:disabled {{ color: {FAINT}; background: {CARD}; border-color: {LINE}; }}
QPushButton#primary {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600;
                       padding: 8px 18px; }}
QPushButton#primary:hover {{ background: {ACCENT_HI}; border-color: {ACCENT_HI}; }}
QPushButton#primary:pressed {{ background: {ACCENT_LO}; border-color: {ACCENT_LO}; }}
QPushButton#primary:disabled {{ background: #2C2C31; border-color: #2C2C31; color: {FAINT}; }}
QPushButton#danger {{ background: rgba(255,69,58,0.12); border: 1px solid rgba(255,69,58,0.42); color: #FF7A70; }}
QPushButton#danger:hover {{ background: rgba(255,69,58,0.2); }}
QPushButton#ghost {{ background: transparent; border: 1px solid {LINE_HI}; }}
QPushButton#ghost:hover {{ background: {FIELD}; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT_HI}; padding: 2px 0; text-align: left; }}
QPushButton#link:hover {{ color: #8DBBFF; }}

QPushButton#seg {{ background: transparent; border: none; border-radius: 4px; padding: 5px 10px; color: {DIM};
                   font-weight: 500; }}
QPushButton#seg:hover {{ color: {TEXT}; }}
QPushButton#seg:checked {{ background: {SEG_ON}; color: {TEXT}; }}
#segBox {{ background: {FIELD}; border: 1px solid {LINE}; border-radius: 6px; }}

QToolButton#iconbtn {{ background: {FIELD}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 6px; }}
QToolButton#iconbtn:hover {{ background: {CARD_HI}; }}
QToolButton#iconbtn:checked {{ background: {SEG_ON}; border-color: #4A4A52; }}
QToolButton#rec {{ background: {FIELD}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 6px; }}
QToolButton#rec:hover {{ background: {CARD_HI}; }}
QToolButton#rec:checked {{ background: rgba(255,59,48,0.16); border-color: rgba(255,59,48,0.6); }}
QToolButton#tab {{ background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0;
                   padding: 6px 2px 6px 2px; color: {DIM}; font-size: 9pt; font-weight: 500; }}
QToolButton#tab:hover {{ color: {TEXT}; }}
QToolButton#tab:checked {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QToolButton#flat {{ background: transparent; border: none; border-radius: 4px; padding: 4px; }}
QToolButton#flat:hover {{ background: rgba(255,255,255,0.06); }}

QLineEdit, QSpinBox {{ background: {FIELD}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 6px 9px;
                       selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}

QComboBox {{ background: {FIELD}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 6px 28px 6px 10px;
             min-height: 18px; }}
QComboBox:hover {{ background: {CARD_HI}; }}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url(__ARROW__); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {CARD_HI}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 4px;
                               selection-background-color: {ACCENT}; selection-color: white; color: {TEXT};
                               outline: none; }}
QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 0 8px; border-radius: 4px; }}

QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {SEG_ON}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {ACCENT}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #F4F4F6;
                              border: 1px solid transparent; }}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}
QSlider::sub-page:horizontal:disabled {{ background: #4A4A52; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: #333338; border-radius: 3px; min-height: 30px; margin: 0 2px; }}
QScrollBar::handle:vertical:hover {{ background: #44444A; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0; }}

QMenu {{ background: {CARD_HI}; border: 1px solid {LINE_HI}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 7px 18px 7px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}

QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #55555C;
                        background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; image: url(__CHECK__); }}

QDialog {{ background: {PANEL}; }}
QProgressBar {{ background: {SEG_ON}; border: none; border-radius: 2px; height: 4px; text-align: center;
                color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
"""
