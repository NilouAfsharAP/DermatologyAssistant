import cv2
import numpy as np
from src.ui.image_viewer import SynchronizedImageViewer
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from .styles import STYLE_SHEET
from src.utils.privacy import PrivacyGuard
from src.engine.processor import prepare_input 

class AnalysisWorker(QThread):
    result_ready = pyqtSignal(tuple)
    error_occurred = pyqtSignal(str)

    def __init__(self, ai_engine, image_path, options=None):
        super().__init__()
        self.ai = ai_engine
        self.path = image_path
        self.guard = PrivacyGuard()
        self.options = options or {}

    def run(self):
        try:
            # 1. Privacy Anonymization & Metadata Stripping
            # (Local-only; metadata is not preserved when reading raw pixels.)
            privacy_on = bool(self.options.get("privacy", True))
            full_res_img = self.guard.anonymize(self.path) if privacy_on else cv2.imread(self.path)
            if full_res_img is None: raise ValueError("File read error.")
            
            # Store original dimensions for perfect Heatmap upscaling
            orig_h, orig_w = full_res_img.shape[:2]

            # 2. Preprocessing (optional toggles)
            tensor, rgb_norm = prepare_input(
                full_res_img,
                hair_removal=bool(self.options.get("hair_removal", True)),
                clahe=bool(self.options.get("clahe", True)),
            )

            # 3. Inference (optional TTA) + Grad-CAM++
            label, conf, topk_list, heatmap_small = self.ai.analyze(
                tensor,
                rgb_norm,
                use_tta=bool(self.options.get("tta", True)),
                topk=int(self.options.get("topk", 3)),
            )

            # 4. High-Res Reconstruction
            # Upscale the Grad-CAM++ mask to original resolution for anatomical accuracy
            heatmap_full_res = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            
            self.result_ready.emit((label, conf, topk_list, heatmap_full_res))
        except Exception as e:
            self.error_occurred.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self, ai_engine):
        super().__init__()
        self.engine = ai_engine
        self.setWindowTitle("DermAssist AI | Clinical Decision Support")
        self.resize(1280, 800)
        self.setStyleSheet(STYLE_SHEET)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        side_layout = QVBoxLayout(sidebar)
        
        logo = QLabel("🩺 DermAssist AI")
        logo.setObjectName("LogoText")
        side_layout.addWidget(logo)

        self.btn_upload = QPushButton("📁 Upload Skin Image")
        self.btn_upload.setObjectName("UploadBtn")
        self.btn_upload.clicked.connect(self.upload_image)
        side_layout.addWidget(self.btn_upload)

        # --- Optional preprocessing toggles ---
        self.options_box = QGroupBox("ANALYSIS OPTIONS")
        self.options_box.setObjectName("OptionsGroup")
        opt_layout = QVBoxLayout(self.options_box)
        opt_layout.setContentsMargins(12, 10, 12, 12)
        opt_layout.setSpacing(6)

        self.cb_hair = QCheckBox("Hair removal")
        self.cb_hair.setToolTip("Remove dark hair-like artifacts via inpainting")
        self.cb_hair.setChecked(True)
        self.cb_clahe = QCheckBox("Contrast enhancement\n(CLAHE)")
        self.cb_clahe.setToolTip("Contrast enhancement (CLAHE)")
        self.cb_clahe.setChecked(True)
        self.cb_tta = QCheckBox("Test-time augmentation\n(TTA)")
        self.cb_tta.setToolTip("Test-time augmentation (TTA)")
        self.cb_tta.setChecked(True)

        opt_layout.addWidget(self.cb_hair)
        opt_layout.addWidget(self.cb_clahe)
        opt_layout.addWidget(self.cb_tta)
        side_layout.addWidget(self.options_box)

        side_layout.addSpacing(20)
        side_layout.addWidget(QLabel("ANALYSIS STATUS", objectName="StepLabel"))
        self.status_indicator = QLabel("⚪ System Ready")
        self.status_indicator.setStyleSheet("color: #94a3b8; padding-left: 20px;")
        side_layout.addWidget(self.status_indicator)
        
        side_layout.addStretch()

        # --- Clear disclaimer + local-only note ---
        side_layout.addWidget(QLabel("DISCLAIMER", objectName="StepLabel"))
        disclaimer = QLabel(
            "⚠️ Not a medical device. For research/education only.\n"
            "Do NOT use this output to diagnose or treat.\n\n"
            "🔒 Local-only: Images are processed on your computer and\n"
            "are not uploaded by the app."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("DisclaimerText")
        side_layout.addWidget(disclaimer)

        privacy_badge = QLabel("🛡️ Privacy Guard: Face anonymization ON")
        privacy_badge.setObjectName("PrivacyBadge")
        side_layout.addWidget(privacy_badge)

        # --- Workspace ---
        workspace = QVBoxLayout()
        workspace.setContentsMargins(20, 20, 20, 20)
        workspace.setSpacing(20)

        # Result Card
        self.result_card = QFrame()
        self.result_card.setObjectName("ResultCard")
        self.result_card.setFixedHeight(120)
        res_layout = QHBoxLayout(self.result_card)
        
        text_info = QVBoxLayout()
        self.diag_label = QLabel("Diagnosis: ---")
        self.diag_label.setObjectName("DiagnosisTitle")
        self.conf_label = QLabel("Confidence: 0%")
        self.conf_label.setStyleSheet("color: #64748b; font-size: 16px;")
        text_info.addWidget(self.diag_label)
        text_info.addWidget(self.conf_label)
        
        res_layout.addLayout(text_info)
        res_layout.addStretch()
        
        self.prog_bar = QProgressBar()
        self.prog_bar.setFixedWidth(200)
        self.prog_bar.setRange(0, 100)
        res_layout.addWidget(self.prog_bar)
        workspace.addWidget(self.result_card)

        # Top-3 predictions panel
        self.top3_card = QFrame()
        self.top3_card.setObjectName("Top3Card")
        self.top3_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top3_layout = QVBoxLayout(self.top3_card)
        top3_layout.setContentsMargins(18, 14, 18, 14)

        top3_title = QLabel("TOP-3 PREDICTIONS")
        top3_title.setObjectName("Top3Title")
        top3_layout.addWidget(top3_title)

        self.top3_table = QTableWidget(3, 2)
        self.top3_table.setObjectName("Top3Table")
        self.top3_table.setHorizontalHeaderLabels(["Class", "Probability"])
        self.top3_table.verticalHeader().setVisible(False)
        self.top3_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.top3_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.top3_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.top3_table.horizontalHeader().setStretchLastSection(True)
        self.top3_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top3_table.setShowGrid(False)
        self.top3_table.setAlternatingRowColors(False)

        self.top3_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.top3_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.top3_table.setWordWrap(True)
        self.top3_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.top3_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        header = self.top3_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(80)

        self._fill_topk_placeholder()

        top3_layout.addWidget(self.top3_table)
        workspace.addWidget(self.top3_card)

        # Viewports
        img_area = QHBoxLayout()
        img_area.setSpacing(15)
        
        v1_container = QWidget()
        v1_container.setObjectName("ViewportContainer")
        v1_layout = QVBoxLayout(v1_container)
        self.original_view = SynchronizedImageViewer()
        self.original_view.setObjectName("Viewport")
        v1_layout.addWidget(QLabel("ORIGINAL LESION"))
        v1_layout.addWidget(self.original_view)
        
        v2_container = QWidget()
        v2_container.setObjectName("ViewportContainer")
        v2_layout = QVBoxLayout(v2_container)
        self.heatmap_view = SynchronizedImageViewer()
        self.heatmap_view.setObjectName("Viewport")
        v2_layout.addWidget(QLabel("AI EXPLANATION (HEATMAP)"))
        v2_layout.addWidget(self.heatmap_view)

        img_area.addWidget(v1_container)
        img_area.addWidget(v2_container)
        workspace.addLayout(img_area)

        # Sync Viewers
        self.original_view.view_changed.connect(self.heatmap_view.sync_view)
        self.heatmap_view.view_changed.connect(self.original_view.sync_view)

        # Action Bar
        action_bar = QHBoxLayout()
        self.btn_reset = QPushButton("🧹 Clear All")
        self.btn_reset.clicked.connect(self.reset_ui)
        self.btn_fit = QPushButton("⟲ Fit to Screen")
        self.btn_fit.clicked.connect(self.fit_views)
        self.btn_fit.setEnabled(False)

        action_bar.addStretch()
        action_bar.addWidget(self.btn_fit)
        action_bar.addWidget(self.btn_reset)
        workspace.addLayout(action_bar)

        h_layout.addWidget(sidebar)
        h_layout.addLayout(workspace)

    def _fill_topk_placeholder(self):
        """Fill the top-k table with placeholders."""
        for r in range(self.top3_table.rowCount()):
            self.top3_table.setItem(r, 0, QTableWidgetItem("---"))
            self.top3_table.setItem(r, 1, QTableWidgetItem("---"))
        self._adjust_top3_table_height()

    def _update_topk_table(self, topk_list):
        """Update top-k table from a list of (label, prob_percent)."""
        # Ensure we always show exactly 3 rows in the UI
        rows = self.top3_table.rowCount()
        for r in range(rows):
            if topk_list and r < len(topk_list):
                cls, prob = topk_list[r]
                self.top3_table.setItem(r, 0, QTableWidgetItem(str(cls)))
                self.top3_table.setItem(r, 1, QTableWidgetItem(f"{float(prob):.1f}%"))
            else:
                self.top3_table.setItem(r, 0, QTableWidgetItem("---"))
                self.top3_table.setItem(r, 1, QTableWidgetItem("---"))
        self._adjust_top3_table_height()

    def _adjust_top3_table_height(self):
        """Size the Top-3 table so all rows are visible without scrollbars."""
        self.top3_table.resizeColumnToContents(1)
        self.top3_table.resizeRowsToContents()

        header = self.top3_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        total_h = self.top3_table.horizontalHeader().height()
        for r in range(self.top3_table.rowCount()):
            total_h += self.top3_table.rowHeight(r)
        total_h += self.top3_table.frameWidth() * 2
        self.top3_table.setFixedHeight(total_h)

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            pixmap = QPixmap(path)
            self.original_view.display_image(pixmap)
            self.heatmap_view.clear_image()
            self.btn_fit.setEnabled(True)
            self._fill_topk_placeholder()

            options = {
                "hair_removal": self.cb_hair.isChecked(),
                "clahe": self.cb_clahe.isChecked(),
                "tta": self.cb_tta.isChecked(),
                "topk": 3,
                "privacy": True,  # kept ON by design (local-only + anonymization)
            }

            self.status_indicator.setText("🟡 Analyzing (TTA Active)..." if options["tta"] else "🟡 Analyzing...")
            self.status_indicator.setStyleSheet("color: #f59e0b; padding-left: 20px;")
            self.start_analysis(path, options)

    def start_analysis(self, path, options):
        self.btn_upload.setEnabled(False)
        self.options_box.setEnabled(False)
        self.worker = AnalysisWorker(self.engine, path, options=options)
        self.worker.result_ready.connect(self.on_analysis_done)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self):
        self.btn_upload.setEnabled(True)
        self.options_box.setEnabled(True)

    def on_analysis_done(self, result):
        label, conf, topk_list, heatmap = result
        self.diag_label.setText(f"Diagnosis: {label}")
        self.conf_label.setText(f"Confidence: {conf:.1f}%")
        self.prog_bar.setValue(int(conf))
        self.status_indicator.setText("🟢 Analysis Complete")
        self.status_indicator.setStyleSheet("color: #4ade80; padding-left: 20px;")

        self._update_topk_table(topk_list)
        
        h, w, c = heatmap.shape
        qimg = QImage(heatmap.data, w, h, heatmap.strides[0], QImage.Format.Format_RGB888).copy()
        self.heatmap_view.display_image(QPixmap.fromImage(qimg))
        self.fit_views()

    def fit_views(self):
        """Unified Fix: Force geometry settle then apply transformation."""
        QApplication.processEvents()
        if self.original_view.has_image() and self.heatmap_view.has_image():
            # Reset scaling to 1:1 before fitting to avoid cumulative math errors
            self.original_view.resetTransform()
            self.heatmap_view.resetTransform()
            
            rect = self.original_view.scene.sceneRect()
            self.original_view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.heatmap_view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            
            # Perfect Sync: Heatmap inherits the final Original view matrix
            self.heatmap_view.setTransform(self.original_view.transform())

    def on_error(self, message):
        self.status_indicator.setText("🔴 Analysis Error")
        self.status_indicator.setStyleSheet("color: #ef4444; padding-left: 20px;")
        QMessageBox.critical(self, "Error", f"Clinical Engine Error: {message}")

    def reset_ui(self):
        self.diag_label.setText("Diagnosis: ---")
        self.conf_label.setText("Confidence: 0%")
        self.prog_bar.setValue(0)
        self._fill_topk_placeholder()
        self.original_view.clear_image()
        self.heatmap_view.clear_image()
        self.btn_fit.setEnabled(False)
        self.status_indicator.setText("⚪ System Ready")
        self.status_indicator.setStyleSheet("color: #94a3b8; padding-left: 20px;")