STYLE_SHEET = """
QMainWindow {
    background-color: #f8fafc;
}

/* Sidebar Navigation */
QFrame#Sidebar {
    background-color: #1e293b;
    border-right: 1px solid #e2e8f0;
}

QLabel#LogoText {
    color: #38bdf8;
    font-size: 22px;
    font-weight: bold;
    padding: 20px;
}

/* Step Indicators */
QLabel#StepLabel {
    color: #94a3b8;
    font-size: 12px;
    font-weight: bold;
    margin-left: 20px;
}

/* Action Buttons */
QPushButton#UploadBtn {
    background-color: #38bdf8;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 15px;
    margin: 10px;
    font-size: 14px;
}

QPushButton#UploadBtn:hover {
    background-color: #0ea5e9;
}

/* Results Display */
QFrame#ResultCard {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
}

QFrame#Top3Card {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
}

QLabel#Top3Title {
    color: #334155;
    font-size: 12px;
    font-weight: bold;
}

QTableWidget#Top3Table {
    border: none;
    background-color: transparent;
    color: #0f172a;
    font-size: 12px;
}

QHeaderView::section {
    background-color: transparent;
    color: #64748b;
    font-weight: bold;
    border: none;
    padding: 4px;
}

QLabel#DiagnosisTitle {
    color: #0f172a;
    font-size: 26px;
    font-weight: bold;
}

/* Image Viewports */
QLabel#Viewport {
    background-color: #f1f5f9;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
}

QWidget#ViewportContainer {
    background-color: #f1f5f9;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px;
}

QGraphicsView#Viewport {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: transparent;
}

/* Sidebar Options */
QGroupBox#OptionsGroup {
    color: #e2e8f0;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #334155;
    border-radius: 12px;
    margin: 10px;
    padding-top: 10px;
}

QGroupBox#OptionsGroup::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}

QCheckBox {
    color: #cbd5e1;
    font-weight: normal;
    padding: 2px 0px;
    spacing: 8px;
}

QLabel#DisclaimerText {
    color: #94a3b8;
    font-size: 11px;
    padding: 10px 20px;
}

QLabel#PrivacyBadge {
    color: #4ade80;
    font-size: 11px;
    padding: 12px 20px 20px 20px;
}
"""