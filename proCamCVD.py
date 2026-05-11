import cv2
import numpy as np

# 1. Setup Camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 2. Setup Projector Window
proj_w, proj_h = 1920, 1080
cv2.namedWindow("Projector", cv2.WINDOW_NORMAL)
cv2.moveWindow("Projector", 2120, 0) # Adjusted for your setup
cv2.waitKey(100) 
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 3. Calibration Parameters
grid_size = (6, 4)
spacing_x, spacing_y = 300, 200
radius = 60
calibrated = False
H = None

# Calculate exactly where the dots are in Projector Space (Source Points)
offset_x = (proj_w - (grid_size[0] - 1) * spacing_x) // 2
offset_y = (proj_h - (grid_size[1] - 1) * spacing_y) // 2

proj_pts = []
for y in range(grid_size[1]):
    for x in range(grid_size[0]):
        proj_pts.append((offset_x + x * spacing_x, offset_y + y * spacing_y))
proj_pts = np.array(proj_pts, dtype=np.float32)


def compensate_red_blind(frame):
    # 1. Split channels
    b, g, r = cv2.split(frame.astype(np.float32))

    # 2. CREATE THE TARGETING MASK
    # Only pixels where Red is significantly stronger than Green/Blue get the glow.
    # This prevents the projector from lighting up the floor or white walls.
    red_mask = (r > 120) & (r > g * 1.2) 
    red_mask = red_mask.astype(np.float32)

    # 3. Calculate the Cyan Glow (Red info shifted to G and B)
    g_new = r * 0.9 
    b_new = r * 0.4
    
    # 4. Apply the mask: Multiply by the mask so only 'Red' areas exist
    g_new = g_new * red_mask
    b_new = b_new * red_mask
    r_new = np.zeros_like(r) # Keep Red at 0 for the projector

    # 5. Merge back
    corrected = cv2.merge([b_new, g_new, r_new])
    return np.clip(corrected, 0, 255).astype(np.uint8)


# Blob detector settings to be more 'forgiving'
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50
params.maxArea = 100000
params.filterByCircularity = False
params.minCircularity = 0.2
detector = cv2.SimpleBlobDetector_create(params)

print("Calibrating... Ensure the dots are clearly visible.")

# Global list to store manual clicks
cam_pts = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(cam_pts) < 4:
        cam_pts.append((x, y))
        print(f"Captured Point {len(cam_pts)}: {x}, {y}")


cv2.namedWindow("What the Camera Sees")
cv2.setMouseCallback("What the Camera Sees", mouse_callback)


def apply_checkerboard_compensation(frame, proj_w, proj_h):
    # 1. Create a checkerboard pattern for the whole screen
    # Smaller 'size' means a denser, more noticeable pattern
    size = 40 
    checker = np.zeros((proj_h, proj_w), dtype=np.uint8)
    # Create the pattern using slicing for speed
    checker[::size*2, ::size*2] = 255
    for i in range(0, proj_h, size*2):
        for j in range(0, proj_w, size*2):
            cv2.rectangle(checker, (j, i), (j+size, i+size), 255, -1)
            cv2.rectangle(checker, (j+size, i+size), (j+size*2, i+size*2), 255, -1)
    
    # Convert checker to BGR so we can project in color if we want (e.g., Bright Yellow)
    checker_bgr = cv2.merge([np.zeros_like(checker), checker, checker]) # Yellow/Cyan pattern

    # 2. Find Red areas in the CAMERA frame
    b, g, r = cv2.split(frame.astype(np.float32))
    # Targeting mask: High Red, Low Green/Blue
    red_mask = (r > 100) & (r > g * 1.3)
    red_mask = (red_mask.astype(np.uint8) * 255)

    # 3. Warp the MASK from Camera space to Projector space
    # This ensures the 'hole' for the pattern matches the cup's position
    warped_mask = cv2.warpPerspective(red_mask, H, (proj_w, proj_h))
    
    # Clean up the mask (remove tiny speckles)
    kernel = np.ones((5,5), np.uint8)
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN, kernel)

    # 4. Apply the mask to the checkerboard
    # The projector stays black EXCEPT where the warped red mask is
    output = cv2.bitwise_and(checker_bgr, checker_bgr, mask=warped_mask)
    
    return output
    

# --- INITIALIZE THE HYBRID MESH ---
proj_w, proj_h = 1920, 1080
grid_base = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)

line_gap = 40        # Adjust this (30-60) to change the density
line_thickness = 5   # This provides the 'Chunky' visibility of the checkerboard
grid_color = (255, 255, 0) # Cyan/Yellow provides the best contrast on red

# Option A: The Chunky Wireframe (Horizontal and Vertical Lines)
for x in range(0, proj_w, line_gap):
    cv2.line(grid_base, (x, 0), (x, proj_h), grid_color, line_thickness)
for y in range(0, proj_h, line_gap):
    cv2.line(grid_base, (0, y), (proj_w, y), grid_color, line_thickness)

# Option B: The 'High-Tech' Box Mesh (Alternative look)
# for i in range(0, proj_h, line_gap):
#     for j in range(0, proj_w, line_gap):
#         cv2.rectangle(grid_base, (j, i), (j + line_gap, i + line_gap), grid_color, 2)



while True:
    ret, frame = cap.read()
    if not ret: break

    if not calibrated:
        # Pre-process for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        # Try to find the grid
        found, corners = cv2.findCirclesGrid(thresh, grid_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector)
        
        # Display the dots to the wall
        display = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
        for pt in proj_pts:
            cv2.circle(display, (int(pt[0]), int(pt[1])), radius, (170, 170, 170), -1)

        if found:
            H, _ = cv2.findHomography(corners, proj_pts)
            calibrated = True
            print("DONE! Switching to Compensation Mode.")
            cv2.destroyWindow("What the Camera Sees") # Clean up UI
        
        cv2.imshow("What the Camera Sees", thresh)



        # Inside the 'if not calibrated' block:
        if len(cam_pts) == 4:
            # Map the 4 clicks to the 4 corners of your projected grid
            # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            proj_corners = np.array([
                proj_pts[0],                # Top Left dot
                proj_pts[grid_size[0]-1],   # Top Right dot
                proj_pts[-1],               # Bottom Right dot
                proj_pts[-grid_size[0]]     # Bottom Left dot
            ], dtype=np.float32)
            
            H, _ = cv2.findHomography(np.array(cam_pts, dtype=np.float32), proj_corners)
            calibrated = True
            print("MANUAL CALIBRATION COMPLETE!")


    else:
        # PROJECTOR MODE
        # corrected_frame = compensate_red_blind(frame)
        # display = cv2.warpPerspective(corrected_frame, H, (proj_w, proj_h))
        
        
        # 1. Identify Red in the CAMERA frame
        b, g, r = cv2.split(frame.astype(np.float32))
        red_mask = (r > 120) & (r > g * 1.4) # Target the cup
        red_mask = (red_mask.astype(np.uint8) * 255)

        # 2. Warp the MASK to Projector Space
        # This is where the magic happens: aligning the mask to the wall
        warped_mask = cv2.warpPerspective(red_mask, H, (proj_w, proj_h))

        # 3. Dilate the mask (The 'Halo' Effect)
        # This makes the mesh slightly larger than the cup so the edges are visible
        kernel = np.ones((7,7), np.uint8)
        warped_mask = cv2.dilate(warped_mask, kernel, iterations=1)

        # 4. Apply the Mesh
        # Bitwise_and acts like a stencil: it only shows the grid where the mask is white
        display = cv2.bitwise_and(grid_base, grid_base, mask=warped_mask)

    cv2.imshow("Projector", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()