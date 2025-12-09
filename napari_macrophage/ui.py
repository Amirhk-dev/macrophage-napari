from qtpy import QtWidgets, QtCore

###### build widget and set style ######
def _widget_stylesheet() -> str:
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
    native = mg_widget.native if hasattr(mg_widget, "native") else mg_widget
    btns = native.findChildren(QtWidgets.QPushButton)
    if btns:
        btns[0].setToolTip(text)
        btns[0].setStatusTip(text)
        btns[0].setWhatsThis(text)
    else:
        native.setToolTip(text)
