"""The dark photo-editor look.

A dark, desaturated surround is not decoration here: judging exposure and
colour against a bright interface genuinely biases what you see, which is why
every serious photo tool ships this way. Everything in the chrome is neutral
grey so the only saturated thing on screen is the photograph.
"""

from __future__ import annotations

#: One accent, used only for focus, selection and the primary action. Anything
#: that competes with the image for attention has to earn it.
ACCENT = "#4a9eff"

BACKGROUND = "#1c1c1e"      # window
SURFACE = "#252528"         # panels
SURFACE_RAISED = "#2e2e32"  # inputs, cards
BORDER = "#3a3a3f"
TEXT = "#e8e8ea"
TEXT_DIM = "#8e8e96"
CANVAS = "#141416"          # behind the image, darkest thing on screen

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}

QLabel {{ background: transparent; }}
QLabel[dim="true"] {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel[heading="true"] {{
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 2px 0 2px 0;
}}

#Panel {{ background: {SURFACE}; }}
#Canvas {{ background: {CANVAS}; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: {SURFACE}; border: none; }}

/* --- inputs ------------------------------------------------------------ */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 8px;
    min-height: 18px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: #4a4a52;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
/* Qt's Fusion style draws proper arrows for combo boxes and spin boxes. A
   CSS-border triangle renders as a grey square here, and simply hiding the
   default leaves no affordance at all, so the native arrows are left alone. */

QComboBox QAbstractItemView {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

/* --- buttons ----------------------------------------------------------- */
QPushButton {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 7px 14px;
}}
QPushButton:hover {{ background: #35353a; }}
QPushButton:pressed {{ background: #202024; }}
QPushButton:disabled {{ color: #5a5a62; background: #232326; }}
QPushButton[primary="true"] {{
    background: {ACCENT};
    border: none;
    color: #06121f;
    font-weight: 600;
    padding: 9px 18px;
}}
QPushButton[primary="true"]:hover {{ background: #5aa9ff; }}
QPushButton[primary="true"]:disabled {{ background: #2f4459; color: #6d7d8d; }}

/* --- sliders ----------------------------------------------------------- */
QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #c8c8ce;
    width: 13px;
    height: 13px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::sub-page:horizontal {{ background: #55555e; border-radius: 2px; }}

/* --- checkboxes -------------------------------------------------------- */
QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE_RAISED};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* --- lists ------------------------------------------------------------- */
QListWidget {{
    background: {SURFACE};
    border: none;
    outline: none;
}}
QListWidget::item {{
    border-radius: 6px;
    margin: 2px 4px;
    padding: 4px;
}}
QListWidget::item:selected {{ background: #33333a; }}
QListWidget::item:hover:!selected {{ background: #2a2a2f; }}

/* --- tabs -------------------------------------------------------------- */
QTabWidget::pane {{ border: none; background: {SURFACE}; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 8px 14px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

/* --- misc -------------------------------------------------------------- */
QProgressBar {{
    background: {SURFACE_RAISED};
    border: none;
    border-radius: 2px;
    height: 4px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: #3d3d44; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #4d4d55; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: #35353c;
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px 7px;
    border-radius: 4px;
}}

QSplitter::handle {{ background: {BACKGROUND}; width: 2px; }}
QSplitter::handle:hover {{ background: {BORDER}; }}
"""


def apply(app) -> None:
    """Apply the theme to a QApplication."""
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    app.setStyle("Fusion")

    # Qt draws a few things (native dialogs, tooltips before styling) from the
    # palette rather than the stylesheet, so both have to agree or file dialogs
    # come up blindingly white in the middle of a dark app.
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#06121f"))
    palette.setColor(QPalette.ToolTipBase, QColor("#35353c"))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#5a5a62"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5a5a62"))
    app.setPalette(palette)

    app.setStyleSheet(STYLESHEET)
