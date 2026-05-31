import cv2
import numpy as np
from face_processor import FaceLipTracker

# --- CONFIGURATION ---
proj_w, proj_h = 1920, 1080
calibrated = False
H = None
manual_cam_pts = []

# Initialize new Tasks-based Tracker
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
        # Pull dense landmark calculations
        tracking_data = tracker.get_lip_points(frame)
        
        if tracking_data is not None:
            all_coords, lip_coords = tracking_data
            cam_h, cam_w = frame.shape[0], frame.shape[1]

            # 1. Rasterize tracking data into a clean binary mask
            cam_lip_mask = np.zeros((cam_h, cam_w), dtype=np.uint8)
            lip_poly = lip_coords[0].astype(np.int32)
            cv2.fillPoly(cam_lip_mask, [lip_poly], 255)

            # 2. Map coordinates safely via spatial homography
            warped_mask = cv2.warpPerspective(cam_lip_mask, H, (proj_w, proj_h))
            warped_mask = warped_mask.astype(np.uint8)

            # 3. Stitch crimson fill onto output canvas
            lipstick_color = np.full((proj_h, proj_w, 3), (50, 50, 200), dtype=np.uint8)
            display = cv2.bitwise_and(lipstick_color, lipstick_color, mask=warped_mask)

            # 4. Render dense tracking matrix on laptop screen for telemetry verification
            for pt in all_coords[0]:
                cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

    cv2.imshow("Projector", display)
    cv2.imshow("Debug View", debug_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()