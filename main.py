import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.engine.inference import DermatologyAI
from src.ui.main_window import MainWindow
from src.ui.styles import STYLE_SHEET


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    root = Path(__file__).resolve().parent
    weights = root / "models" / "skin_classifier.pth"
    engine = DermatologyAI(weights)

    window = MainWindow(engine)
    window.show()
    sys.exit(app.exec())
