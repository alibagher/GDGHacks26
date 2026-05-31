import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceLipTracker:
    def __init__(self):
        # 1. Configure the modern MediaPipe Task options
        # Points directly to the downloaded model bundle file in your folder
        model_path = 'face_landmarker.task'
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # FIX: Removed 'output_face_landmarks=True' as it is implicit
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False, 
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        
        # 2. Build the detector object
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Explicit landmark indices tracking the tight outer vermilion border of the lips
        self.OUTER_LIP_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            308, 415, 310, 311, 312, 13, 82, 81, 80, 191
        ]
        print("MediaPipe Tasks Dense FaceMesh Engine Initialized!")
    def get_lip_points(self, frame):
        h, w, _ = frame.shape
        
        # Convert OpenCV's BGR layout to MediaPipe's Image wrapper format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference synchronously
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None

        face_landmarks = detection_result.face_landmarks[0]
        
        # 1. Parse all 468+ points for full framework Debug View
        all_pts = []
        for landmark in face_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            all_pts.append([cx, cy])
            
        # 2. Isolate target lips mask points
        lip_pts = []
        for idx in self.OUTER_LIP_INDICES:
            landmark = face_landmarks[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            lip_pts.append([cx, cy])
            
        return np.array([all_pts], dtype=np.float32), np.array([lip_pts], dtype=np.float32)