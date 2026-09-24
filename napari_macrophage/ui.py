"""Qt styling helpers and reusable widget primitives (stylesheet, wheel filter,
collapsible section) used by the Macrophage Tools dock."""

from qtpy import QtCore, QtWidgets


###### build widget and set style ######
def _widget_stylesheet() -> str:
    """Return the shared dark-theme Qt stylesheet used by every plugin widget."""
    return """
    QWidget { font-size: 10pt; }
    QGroupBox {
        font-size: 10pt;
        font-weight: normal;
        margin-top: 6px;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px 2px 2px 2px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 4px;
        padding: 2 2px;
        font-weight: 600;
    }
    QPushButton {
        font-size: 10pt;
        padding: 2px 2px;      
        min-height: 14px;     
        border: 1px solid #666;
        border-radius: 3px;
        background-color: #363636;
        color: #ffffff;
    }
    QPushButton:hover {
        background-color: #454545;
    }
    QPushButton:disabled {
        background-color: #2a2a2a;
        color: #888888;
        border-color: #555;
    }
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        font-size: 10pt;
        padding: 3px 3px;     
    }
    """

def _set_call_button_tooltip(mg_widget, text: str):
    """Set ``text`` as the tooltip on ``mg_widget``'s primary call button."""
    native = mg_widget.native if hasattr(mg_widget, "native") else mg_widget
    btns = native.findChildren(QtWidgets.QPushButton)
    if btns:
        btns[0].setToolTip(text)
        btns[0].setStatusTip(text)
        btns[0].setWhatsThis(text)
    else:
        native.setToolTip(text)


class _NoWheelFilter(QtCore.QObject):
    """Event filter that eats wheel events so scrolling doesn't change values."""

    def eventFilter(self, obj, event):
        """Swallow wheel events on the watched widgets so they don't change values."""
        if event.type() == QtCore.QEvent.Wheel:
            event.ignore()
            return True
        return False


_no_wheel_filter = _NoWheelFilter()


def _disable_wheel_on_inputs(widget):
    """Prevent mouse-wheel from changing spinbox/combobox values inside ``widget``."""
    native = widget.native if hasattr(widget, "native") else widget
    for w in native.findChildren(QtWidgets.QAbstractSpinBox):
        w.setFocusPolicy(QtCore.Qt.StrongFocus)
        w.installEventFilter(_no_wheel_filter)
    for w in native.findChildren(QtWidgets.QComboBox):
        w.setFocusPolicy(QtCore.Qt.StrongFocus)
        w.installEventFilter(_no_wheel_filter)


class CollapsibleSection(QtWidgets.QWidget):
    """A group-box-like container with a clickable header that hides its content.

    The header is a full-width tool button that shows a right-arrow when
    collapsed and a down-arrow when expanded. Content is added via
    :meth:`setContentLayout`. Sections behave independently — clicking one has
    no effect on the others.
    """

    def __init__(self, title: str, parent=None, expanded: bool = False):
        """Build the header + content pair. ``expanded`` sets the initial state."""
        super().__init__(parent)

        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QToolButton {"
            " text-align: left;"
            " padding: 4px 6px;"
            " border: 1px solid #444;"
            " border-radius: 4px;"
            " background-color: #2a2a2a;"
            " color: #ffffff;"
            " font-weight: 600;"
            "}"
            "QToolButton:hover { background-color: #363636; }"
            "QToolButton:checked { background-color: #363636; }"
        )
        self._toggle.clicked.connect(self._on_toggle)

        self._content = QtWidgets.QWidget(self)
        self._content.setVisible(expanded)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def setContentLayout(self, content_layout: QtWidgets.QLayout) -> None:
        """Install ``content_layout`` as the collapsible content."""
        old = self._content.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)  # detach any previous layout
        self._content.setLayout(content_layout)

    def setExpanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse the section."""
        self._toggle.setChecked(expanded)
        self._on_toggle(expanded)

    def isExpanded(self) -> bool:
        """Return ``True`` if the section is currently expanded."""
        return self._toggle.isChecked()

    def _on_toggle(self, checked: bool) -> None:
        """Header click handler: flip the arrow and toggle content visibility."""
        self._toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._content.setVisible(checked)
