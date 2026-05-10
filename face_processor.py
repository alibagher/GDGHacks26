import dlib
import numpy as np
import cv2

class FaceLipTracker:
    def __init__(self):
        # Initialize dlib's face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Load the predictor (Ensure this .dat file is in your folder!)
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("Dlib Tracker Initialized Successfully!")
        except Exception as e:
            print(f"FATAL ERROR: shape_predictor_68_face_landmarks.dat not found. {e}")
            self.predictor = None

        # Dlib indices for the outer boundary of the lips
        self.OUTER_LIP_INDICES = list(range(48, 60))

    def get_lip_points(self, frame):
        if self.predictor is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None

        # Process the first face (the printout)
        shape = self.predictor(gray, faces[0])
        
        # All 68 points for the Debug View
        all_pts = []
        for i in range(0, 68):
            all_pts.append([shape.part(i).x, shape.part(i).y])
            
        # Specific lip points for the Projector Mask
        lip_pts = []
        for i in self.OUTER_LIP_INDICES:
            lip_pts.append([shape.part(i).x, shape.part(i).y])
            
        return np.array([all_pts], dtype=np.float32), np.array([lip_pts], dtype=np.float32)