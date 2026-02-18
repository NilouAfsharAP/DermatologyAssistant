from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent

class SynchronizedImageViewer(QGraphicsView):
    view_changed = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # We keep a reference to this item throughout the app life
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self._is_syncing = False
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.horizontalScrollBar().valueChanged.connect(self.emit_view_change)
        self.verticalScrollBar().valueChanged.connect(self.emit_view_change)

    def display_image(self, pixmap: QPixmap):
        """Loads a pixmap and fits it to view initially."""
        if pixmap.isNull():
            return
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view_custom()
        self.setBackgroundBrush(Qt.GlobalColor.black)

    def clear_image(self):
        """Safely clears the image without deleting the C++ object."""
        self.pixmap_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF())
        self.resetTransform() # Reset zoom/pan
        self.setBackgroundBrush(Qt.GlobalColor.transparent)

    def has_image(self):
        return self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull()

    def fit_in_view_custom(self):
        """Utility to center the image."""
        if self.has_image():
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if not self.has_image(): return
        self._is_syncing = True
        
        zoom_in = 1.15
        zoom_out = 1 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        
        # Check scale limits
        if (self.transform().m11() < 0.1 and factor < 1) or (self.transform().m11() > 20 and factor > 1):
            self._is_syncing = False
            return

        self.scale(factor, factor)
        self.emit_view_change()
        self._is_syncing = False

    def emit_view_change(self):
        if not self._is_syncing and self.has_image():
            visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            self.view_changed.emit(visible_rect)

    def sync_view(self, rect: QRectF):
        if not self.has_image() or self._is_syncing: return
        self._is_syncing = True
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._is_syncing = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_in_view_custom()