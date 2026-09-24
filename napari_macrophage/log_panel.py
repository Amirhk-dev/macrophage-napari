"""In-plugin log panel that mirrors napari notifications.

Constructing a :class:`LogPanel` subscribes to
``napari.utils.notifications.notification_manager`` so every ``show_info``,
``show_warning``, and ``show_error`` call is timestamped and appended to a
scrollable text widget. First construction also monkey-patches napari's
bottom-of-window popup to a no-op, so all messages appear only in the panel.
"""

import datetime
import html as _html

from napari.utils.notifications import notification_manager
from qtpy import QtCore, QtGui, QtWidgets

_SEVERITY_COLORS = {
    "info": "#dcdcdc",
    "warning": "#ffcc66",
    "error": "#ff6b6b",
    "debug": "#8fa1b3",
    "none": "#dcdcdc",
}

_popups_suppressed = False


def _suppress_native_popups() -> None:
    """Replace napari's bottom-of-window notification popup with a no-op."""
    global _popups_suppressed
    if _popups_suppressed:
        return
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification
        NapariQtNotification.show_notification = staticmethod(lambda notification: None)
        _popups_suppressed = True
    except Exception:
        pass


class LogPanel(QtWidgets.QWidget):
    """Scrollable in-plugin log that mirrors napari notifications."""

    _message_signal = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        """Build the widget and start mirroring napari notifications into it."""
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        title = QtWidgets.QLabel("Log")
        title.setStyleSheet("font-weight: bold;")
        clear_btn = QtWidgets.QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.clear)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(500)
        self._text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self._text.setMinimumHeight(120)
        self._text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        f = QtGui.QFont()
        f.setStyleHint(QtGui.QFont.TypeWriter)
        f.setPointSize(9)
        self._text.setFont(f)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #1e1e1e;"
            " color: #dcdcdc;"
            " border: 1px solid #444;"
            "}"
        )
        layout.addWidget(self._text)

        self._message_signal.connect(self._append, QtCore.Qt.QueuedConnection)

        try:
            notification_manager.notification_ready.connect(self._on_notification)
        except Exception:
            pass

        _suppress_native_popups()

        self.destroyed.connect(lambda *_: self._disconnect())

    def clear(self) -> None:
        """Empty the log."""
        self._text.clear()

    def _disconnect(self) -> None:
        """Detach from the notification manager (called on widget destruction)."""
        try:
            notification_manager.notification_ready.disconnect(self._on_notification)
        except Exception:
            pass

    def _on_notification(self, notification) -> None:
        """Notification-manager callback; forwards to the widget's Qt signal."""
        try:
            severity = str(getattr(notification, "severity", "info")).lower()
        except Exception:
            severity = "info"
        try:
            message = str(getattr(notification, "message", notification))
        except Exception:
            message = str(notification)
        # Normalize enum-style strings like "NotificationSeverity.INFO"
        if "." in severity:
            severity = severity.rsplit(".", 1)[-1]
        self._message_signal.emit(severity, message)

    def _append(self, severity: str, message: str) -> None:
        """Append a timestamped, color-coded message to the log widget (main thread)."""
        color = _SEVERITY_COLORS.get(severity, "#dcdcdc")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        esc_msg = _html.escape(message).replace("\n", "<br>")
        line = (
            f'<span style="color:#888">[{ts}]</span> '
            f'<b style="color:{color}">{severity.upper()}</b> '
            f'<span style="color:{color}">{esc_msg}</span>'
        )
        self._text.appendHtml(line)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())
