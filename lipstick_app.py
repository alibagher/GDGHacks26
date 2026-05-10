import cv2
import numpy as np
from face_processor import FaceLipTracker

# --- CONFIGURATION ---
proj_w, proj_h = 1920, 1080
calibrated = False
H = None
manual_cam_pts = []

tracker = FaceLipTracker()

cv2.namedWindow("Projector", cv2.WINDOW_NORMAL)
cv2.moveWindow("Projector", 2230, 1090) 
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow("Debug View")

def mouse_callback(event, x, y, flags, param):
    global manual_cam_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_cam_pts) < 4:
        manual_cam_pts.append((x, y))

cv2.setMouseCallback("Debug View", mouse_callback)

proj_pts = np.array([[400, 300], [1520, 300], [1520, 780], [400, 780]], dtype=np.float32)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
    debug_frame = frame.copy()

    if not calibrated:
        for pt in proj_pts:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 20, (255, 255, 255), -1)
        if len(manual_cam_pts) == 4:
            H, _ = cv2.findHomography(np.array(manual_cam_pts, dtype=np.float32), proj_pts)
            calibrated = True
    
    else:
        # 1. TRACK (Sensing)
        tracking_data = tracker.get_lip_points(frame)
        
        if tracking_data is not None:
            all_coords, lip_coords = tracking_data
            
            # 2. CREATE CAMERA-SPACE MASK (This is your 'red_mask' equivalent)
            # We create a black image the EXACT size of your camera feed
            cam_lip_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            # Convert Dlib points to integers for OpenCV drawing
            lip_poly = lip_coords[0].astype(np.int32)
            
            # Draw the 'lipstick' area on the camera-sized mask
            cv2.fillPoly(cam_lip_mask, [lip_poly], 255)

            # 3. THE CVD ALIGNMENT STEP (The Magic)
            # We warp the ENTIRE camera-sized mask to the projector-sized canvas
            # using the H matrix from your 4-point calibration.
            warped_mask = cv2.warpPerspective(cam_lip_mask, H, (proj_w, proj_h))

            # 4. DILATE (Your 'Halo' effect for better coverage)
            kernel = np.ones((5,5), np.uint8)
            warped_mask = cv2.dilate(warped_mask, kernel, iterations=1)

            # 5. APPLY COLOR
            # This mimics your 'bitwise_and' with the grid_base
            lipstick_color = np.full((proj_h, proj_w, 3), (50, 50, 200), dtype=np.uint8)
            display = cv2.bitwise_and(lipstick_color, lipstick_color, mask=warped_mask)

            # Debug View (laptop screen)
            for pt in all_coords[0]:
                cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

    cv2.imshow("Projector", display)
    cv2.imshow("Debug View", debug_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()