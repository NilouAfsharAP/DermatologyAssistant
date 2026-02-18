import cv2
import os

class PrivacyGuard:
    def __init__(self):
        cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade_path)

    def anonymize(self, img_path):
        """Strips metadata and blurs faces."""
        img = cv2.imread(img_path)
        if img is None: return None
        
        # Metadata is automatically stripped because we only use the raw pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            # Heavy blur to ensure unidentifiable status
            img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (101, 101), 30)
        return img