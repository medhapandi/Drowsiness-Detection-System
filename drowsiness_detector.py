# drowsiness_detector.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import importlib

# Try to import optional face detectors
FaceDetector = None
try:
    fd_mod = importlib.import_module('utils.face_detector')
    FaceDetector = getattr(fd_mod, 'FaceDetector', None)
except Exception:
    try:
        fd_mod = importlib.import_module('utils.face_detector_mediapipe')
        FaceDetector = getattr(fd_mod, 'FaceDetector', None)
    except Exception:
        FaceDetector = None


class DrowsinessDetector:
    def __init__(self, model_path='model/eye_model.h5'):
        # Haar cascade fallbacks
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Try to load the CNN model, but don't fail if it's not available
        self.eye_model = None
        try:
            self.eye_model = keras.models.load_model(model_path)
            print("CNN model loaded successfully")
        except Exception:
            print("No CNN model found. Using basic detection only.")

        # Try to instantiate a FaceDetector if available
        self.face_detector = None
        if FaceDetector is not None:
            try:
                self.face_detector = FaceDetector()
                print(f"Using face detector: {FaceDetector.__module__}")
            except Exception as e:
                print("FaceDetector available but failed to initialize:", e)
                self.face_detector = None

        # State variables
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.alert_threshold = 15  # frames

    def _predict_eyes_closed(self, left_eye, right_eye):
        """Return True if eyes are closed based on CNN if available, else simple brightness heuristic."""
        if self.eye_model is not None and left_eye is not None and right_eye is not None:
            # model expects (24,24,1)
            try:
                l = left_eye.reshape(1, 24, 24, 1).astype('float32') / 255.0
                r = right_eye.reshape(1, 24, 24, 1).astype('float32') / 255.0
                # average prediction across both eyes
                p1 = float(self.eye_model.predict(l, verbose=0)[0][0])
                p2 = float(self.eye_model.predict(r, verbose=0)[0][0])
                prob = (p1 + p2) / 2.0
                # model trains closed=1, open=0 so higher => closed
                return prob > 0.5
            except Exception:
                pass

        # Fallback heuristic: if both eye crops are very dark, treat as closed
        vals = []
        for e in (left_eye, right_eye):
            if e is not None:
                vals.append(np.mean(e))
        if not vals:
            # no eye data — assume closed
            return True
        mean_brightness = float(np.mean(vals))
        return mean_brightness < 50.0

    def detect_drowsiness(self, frame):
        """Main drowsiness detection function."""
        results = {
            'eyes_closed': False,
            'yawning': False,
            'eye_aspect_ratio': 0.0,
            'mouth_open_ratio': 0.0
        }

        # Prefer using an advanced face detector (dlib or mediapipe) if available
        if self.face_detector is not None:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                # use first face
                face = faces[0]
                left_eye, right_eye = self.face_detector.extract_eyes(frame, face)

                eyes_closed = self._predict_eyes_closed(left_eye, right_eye)
                if eyes_closed:
                    self.eye_closed_frames += 1
                else:
                    self.eye_closed_frames = 0

                # yawning detection via face detector if available
                try:
                    mar = float(self.face_detector.detect_yawn(frame, face))
                except Exception:
                    mar = 0.0

                results['eye_aspect_ratio'] = 0.0
                results['mouth_open_ratio'] = mar

                if mar > 0.6:
                    self.yawn_frames += 1
                else:
                    self.yawn_frames = 0

        else:
            # Haar-cascade fallback (simple and less reliable)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)

                    if len(eyes) < 1:
                        self.eye_closed_frames += 1
                    else:
                        self.eye_closed_frames = 0

                    mouth_roi = roi_gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
                    if mouth_roi.size > 0:
                        mouth_brightness = np.mean(mouth_roi)
                        if mouth_brightness > 100:
                            self.yawn_frames += 1
                        else:
                            self.yawn_frames = 0

        # Set results based on state
        results['eyes_closed'] = self.eye_closed_frames > self.alert_threshold
        results['yawning'] = self.yawn_frames > self.alert_threshold

        return results