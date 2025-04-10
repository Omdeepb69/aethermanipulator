import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
import sys
import time

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    # PyOpenGL_accelerate is implicitly used if installed
except ImportError:
    print("ERROR: PyOpenGL or PyOpenGL_accelerate not installed.")
    print("Please install them:")
    print("pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

try:
    import pywavefront
except ImportError:
    print("ERROR: pywavefront not installed.")
    print("Please install it:")
    print("pip install pywavefront")
    sys.exit(1)

# --- Configuration ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "AetherManipulator"
DEFAULT_MODEL_SCALE = 1.0
MODEL_ROTATION_SPEED = 0.8
MODEL_TRANSLATION_SPEED = 1.5
MODEL_SCALE_SPEED = 0.05
UI_COLOR = (0, 200, 0) # Green
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_SCALE = 0.7
UI_THICKNESS = 2

# --- Global State ---
interaction_mode = "TRANSLATE" # TRANSLATE, ROTATE, SCALE
model_translation = [0.0, 0.0, -5.0] # Initial translation (X, Y, Z)
model_rotation = [0.0, 0.0, 0.0] # Initial rotation (X, Y, Z degrees)
model_scale = DEFAULT_MODEL_SCALE

# Hand tracking state
last_hand_centers = [None, None] # Store center positions for delta calculations (index 0 for first hand, 1 for second)
initial_pinch_distance = None # For scaling
last_gesture_time = 0 # Debounce gestures

# --- OpenGL Rendering ---
scene = None
light_ambient = [0.2, 0.2, 0.2, 1.0]
light_diffuse = [0.8, 0.8, 0.8, 1.0]
light_specular = [0.5, 0.5, 0.5, 1.0]
light_position = [2.0, 2.0, 5.0, 1.0] # Positional light

def init_opengl(width, height):
    """Initializes OpenGL context settings."""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    glShadeModel(GL_SMOOTH)
    glClearColor(0.1, 0.1, 0.15, 0.0) # Dark background
    glClearDepth(1.0)

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def load_model(obj_file_path):
    """Loads an OBJ model using pywavefront."""
    global scene, model_scale, model_translation
    try:
        loaded_scene = pywavefront.Wavefront(obj_file_path, collect_faces=True, parse=True)

        # Calculate initial scale and center
        all_vertices = np.array([v for mesh in loaded_scene.mesh_list for v in mesh.vertices])
        if len(all_vertices) == 0:
            print(f"Warning: Model '{obj_file_path}' contains no vertices.")
            return None # Return None if model is empty

        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        center = (min_coords + max_coords) / 2.0
        size = np.max(max_coords - min_coords)

        # Normalize scale and position
        scale_factor = DEFAULT_MODEL_SCALE / max(size, 1e-6) # Avoid division by zero
        model_scale = scale_factor
        model_translation = [-center[0] * scale_factor, -center[1] * scale_factor, -5.0] # Center and move back

        print(f"Loaded model: {os.path.basename(obj_file_path)}")
        print(f" - Vertices: {len(all_vertices)}")
        print(f" - Initial Center: {center}")
        print(f" - Initial Size: {size}")
        print(f" - Applied Scale: {model_scale}")
        print(f" - Initial Translation: {model_translation}")

        return loaded_scene
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {obj_file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load model '{obj_file_path}': {e}")
        return None

def draw_model(current_scene):
    """Renders the loaded OBJ model."""
    if not current_scene:
        return

    glPushMatrix()
    # Apply transformations
    glTranslatef(model_translation[0], model_translation[1], model_translation[2])
    glRotatef(model_rotation[0], 1.0, 0.0, 0.0) # Rotate around X
    glRotatef(model_rotation[1], 0.0, 1.0, 0.0) # Rotate around Y
    glRotatef(model_rotation[2], 0.0, 0.0, 1.0) # Rotate around Z
    glScalef(model_scale, model_scale, model_scale)

    # Render using pywavefront's drawing capabilities
    for mesh in current_scene.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_index in face:
                if vertex_index < len(current_scene.vertices):
                    vertex = current_scene.vertices[vertex_index]
                    if len(vertex) >= 3: # Ensure vertex has at least 3 components (x, y, z)
                        # Check for normals (optional but recommended for lighting)
                        if vertex_index < len(current_scene.normals):
                             normal = current_scene.normals[vertex_index]
                             if len(normal) == 3:
                                 glNormal3fv(normal)
                        # Set vertex color (if available, otherwise use default)
                        # pywavefront doesn't directly expose vertex colors easily here
                        # We'll rely on glColorMaterial and potentially material colors if set
                        glVertex3fv(vertex[:3]) # Use only x, y, z
        glEnd()

    glPopMatrix()

def render_gl_scene(width, height, current_scene):
    """Renders the OpenGL scene and returns it as a NumPy array."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Position the light relative to the view
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    # Draw the model with current transformations
    draw_model(current_scene)

    # Read pixels back from OpenGL buffer
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE) # Read as BGR for OpenCV

    # Reshape into NumPy array
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    # OpenGL renders bottom-up, OpenCV expects top-down
    image = cv2.flip(image, 0)
    return image

# --- MediaPipe Hand Tracking ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=0, # 0 for faster performance
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2)

# --- Gesture Recognition Helpers ---
def get_landmarks_list(hand_landmarks, frame_shape):
    """Converts MediaPipe landmarks to a list of (x, y, z) pixel coordinates."""
    h, w = frame_shape[:2]
    landmarks = []
    if hand_landmarks:
        for lm in hand_landmarks.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))
    return landmarks

def get_hand_center(landmarks_list):
    """Calculates the approximate center (centroid) of the hand landmarks."""
    if not landmarks_list:
        return None
    x_coords = [lm[0] for lm in landmarks_list]
    y_coords = [lm[1] for lm in landmarks_list]
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    # Use average Z of wrist and MCP joints for a more stable depth estimate
    z_coords = [landmarks_list[i][2] for i in [0, 5, 9, 13, 17] if i < len(landmarks_list)]
    center_z = np.mean(z_coords) if z_coords else 0.0
    return (center_x, center_y, center_z)

def is_fist(landmarks_list):
    """Determines if a hand is likely in a fist gesture."""
    if not landmarks_list or len(landmarks_list) < 21:
        return False

    # Check if fingertips (8, 12, 16, 20) are close to the palm center/base
    # Using wrist (0) and middle finger MCP (9) as reference points
    try:
        wrist = np.array(landmarks_list[0][:2]) # Use only x, y for distance check
        mcp_9 = np.array(landmarks_list[9][:2])
        palm_center = (wrist + mcp_9) / 2

        tip_indices = [8, 12, 16, 20]
        pip_indices = [6, 10, 14, 18] # Proximal Interphalangeal joints

        # Calculate average distance from fingertips to palm center
        avg_tip_dist = np.mean([np.linalg.norm(np.array(landmarks_list[i][:2]) - palm_center) for i in tip_indices])

        # Calculate average distance from PIP joints to palm center (for reference)
        avg_pip_dist = np.mean([np.linalg.norm(np.array(landmarks_list[i][:2]) - palm_center) for i in pip_indices])

        # Heuristic: If tips are closer to the palm center than PIP joints (or close to it), it's likely a fist
        # Add a check for thumb tip (4) being close to index MCP (5) or middle MCP (9)
        thumb_tip = np.array(landmarks_list[4][:2])
        index_mcp = np.array(landmarks_list[5][:2])
        thumb_dist_to_index = np.linalg.norm(thumb_tip - index_mcp)

        # Thresholds (may need tuning)
        fist_tip_threshold_ratio = 0.9 # Tips should be closer than PIPs
        fist_thumb_threshold = np.linalg.norm(np.array(landmarks_list[5][:2]) - np.array(landmarks_list[17][:2])) * 0.6 # Thumb close to index base

        # print(f"Debug Fist: Avg Tip Dist: {avg_tip_dist:.2f}, Avg PIP Dist: {avg_pip_dist:.2f}, Thumb Dist: {thumb_dist_to_index:.2f}")

        return avg_tip_dist < avg_pip_dist * fist_tip_threshold_ratio and thumb_dist_to_index < fist_thumb_threshold

    except IndexError:
        # Handle cases where landmarks might be missing temporarily
        return False
    except Exception as e:
        print(f"Error in is_fist: {e}")
        return False


def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def calculate_2d_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# --- Main Application Logic ---
def main(model_path):
    global scene, interaction_mode, model_translation, model_rotation, model_scale
    global last_hand_centers, initial_pinch_distance, last_gesture_time

    # --- Initialization ---
    print("Initializing AetherManipulator...")

    # Load Model
    scene = load_model(model_path)
    if scene is None:
        # Attempt to load a default/fallback model if primary fails?
        # For now, just exit if the specified model fails.
        print("Exiting due to model loading failure.")
        return

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {actual_width}x{actual_height}")

    # Create OpenCV window and initialize OpenGL
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Use NORMAL for potential resizing
    cv2.resizeWindow(WINDOW_NAME, actual_width, actual_height)
    # Crucially, init OpenGL *after* the window context might be available (though not directly bound)
    # We render offscreen anyway, so direct binding isn't strictly necessary here.
    init_opengl(actual_width, actual_height)

    print("Initialization complete. Starting real-time loop...")

    # --- Main Loop ---
    prev_time = time.time()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / delta_time if delta_time > 0 else 0

        # Flip the frame horizontally for a later selfie-view display
        # Process the original frame for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Prepare flipped frame for drawing
        frame = cv2.flip(frame, 1)

        # Process hand landmarks
        detected_hands = []
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on the *flipped* frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                landmarks_list = get_landmarks_list(hand_landmarks, frame.shape)
                if landmarks_list:
                    hand_center = get_hand_center(landmarks_list)
                    # Store hand data (landmarks, center, handedness)
                    handedness = results.multi_handedness[i].classification[0].label
                    detected_hands.append({
                        "landmarks": landmarks_list,
                        "center": hand_center,
                        "handedness": handedness,
                        "is_fist": is_fist(landmarks_list)
                    })

        # --- Gesture Mapping and State Update ---
        num_hands = len(detected_hands)
        current_hand_centers = [None, None]
        gesture_debounce_time = 0.2 # Seconds to wait before changing mode

        # Determine Interaction Mode based on number of hands and gestures
        new_mode = interaction_mode # Default to current mode
        if num_hands == 1:
            hand = detected_hands[0]
            current_hand_centers[0] = hand["center"]
            if hand["is_fist"]:
                new_mode = "ROTATE"
            else: # Open hand
                new_mode = "TRANSLATE"
        elif num_hands == 2:
            new_mode = "SCALE"
            # Ensure consistent ordering (e.g., left hand first if possible)
            hand0 = detected_hands[0] if detected_hands[0]['handedness'] == 'Left' else detected_hands[1]
            hand1 = detected_hands[1] if detected_hands[0]['handedness'] == 'Left' else detected_hands[0]
            current_hand_centers[0] = hand0["center"]
            current_hand_centers[1] = hand1["center"]
        else: # No hands or more than 2 (ignore > 2 for now)
            new_mode = interaction_mode # Keep last mode briefly
            last_hand_centers = [None, None] # Reset tracking history
            initial_pinch_distance = None

        # Apply mode change with debounce
        if new_mode != interaction_mode and (current_time - last_gesture_time > gesture_debounce_time):
             interaction_mode = new_mode
             print(f"Mode changed to: {interaction_mode}")
             last_gesture_time = current_time
             # Reset state specific to the *previous* mode if needed
             if interaction_mode != "SCALE":
                 initial_pinch_distance = None # Reset pinch distance if not scaling
             if interaction_mode != "TRANSLATE" and interaction_mode != "ROTATE":
                 last_hand_centers = [None, None] # Reset single hand tracking


        # --- Apply Transformations based on Mode ---
        if interaction_mode == "TRANSLATE" and num_hands == 1:
            if last_hand_centers[0] and current_hand_centers[0]:
                delta_x = current_hand_centers[0][0] - last_hand_centers[0][0]
                delta_y = current_hand_centers[0][1] - last_hand_centers[0][1]
                # Use Z from landmarks for depth translation (experimental)
                # delta_z = (current_hand_centers[0][2] - last_hand_centers[0][2]) * 5.0 # Adjust sensitivity

                # Map screen coordinates (pixels) to model coordinates
                # Adjust sensitivity based on window size
                trans_sensitivity_x = MODEL_TRANSLATION_SPEED / actual_width
                trans_sensitivity_y = MODEL_TRANSLATION_SPEED / actual_height

                model_translation[0] += delta_x * trans_sensitivity_x
                model_translation[1] -= delta_y * trans_sensitivity_y # Invert Y
                # model_translation[2] += delta_z # Add Z translation

            last_hand_centers[0] = current_hand_centers[0]
            last_hand_centers[1] = None # Ensure second hand history is clear

        elif interaction_mode == "ROTATE" and num_hands == 1:
            if last_hand_centers[0] and current_hand_centers[0]:
                delta_x = current_hand_centers[0][0] - last_hand_centers[0][0]
                delta_y = current_hand_centers[0][1] - last_hand_centers[0][1]

                # Map horizontal movement to Y-axis rotation, vertical to X-axis rotation
                model_rotation[1] += delta_x * MODEL_ROTATION_SPEED # Yaw
                model_rotation[0] += delta_y * MODEL_ROTATION_SPEED # Pitch
                # Keep angles reasonable (optional)
                # model_rotation[0] = max(-90, min(90, model_rotation[0]))

            last_hand_centers[0] = current_hand_centers[0]
            last_hand_centers[1] = None

        elif interaction_mode == "SCALE" and num_hands == 2:
            # Use distance between index finger tips (landmark 8)
            try:
                index_tip1 = detected_hands[0]["landmarks"][8]
                index_tip2 = detected_hands[1]["landmarks"][8]
                current_pinch_distance = calculate_2d_distance(index_tip1, index_tip2)

                if initial_pinch_distance is None:
                    initial_pinch_distance = current_pinch_distance # Capture starting distance
                elif initial_pinch_distance > 1e-6: # Avoid division by zero
                    scale_factor_change = current_pinch_distance / initial_pinch_distance
                    # Apply smooth scaling with a limit to prevent sudden jumps
                    scale_change_clamped = max(0.8, min(1.2, scale_factor_change))  # Limit to 20% change per frame
                    model_scale *= scale_change_clamped
                    
                    # Keep scale within reasonable bounds
                    model_scale = max(0.1, min(10.0, model_scale))
                    
                    # Update initial distance for next frame (smooths movement)
                    initial_pinch_distance = current_pinch_distance
            except (IndexError, KeyError) as e:
                print(f"Error in SCALE mode: {e}")
                
            last_hand_centers[0] = current_hand_centers[0]
            last_hand_centers[1] = current_hand_centers[1]

        # --- Render OpenGL Scene ---
        # Create the 3D visualization using OpenGL
        gl_frame = render_gl_scene(actual_width, actual_height, scene)
        
        # Combine camera feed and OpenGL rendering
        # Use alpha blending for a semi-transparent overlay effect
        alpha = 0.7  # OpenGL visualization opacity
        beta = 1.0 - alpha  # Camera feed opacity
        
        # Resize frames if dimensions don't match
        if gl_frame.shape != frame.shape:
            gl_frame = cv2.resize(gl_frame, (frame.shape[1], frame.shape[0]))
        
        # Overlay the GL visualization on the camera feed
        combined_frame = cv2.addWeighted(frame, beta, gl_frame, alpha, 0)

        # --- Draw UI Elements ---
        # Draw mode indicator
        mode_text = f"Mode: {interaction_mode}"
        cv2.putText(combined_frame, mode_text, (20, 30), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        # Draw FPS counter
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(combined_frame, fps_text, (20, 60), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        # Draw transformation values
        trans_text = f"Position: X={model_translation[0]:.1f} Y={model_translation[1]:.1f} Z={model_translation[2]:.1f}"
        cv2.putText(combined_frame, trans_text, (20, actual_height - 90), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        rot_text = f"Rotation: X={model_rotation[0]:.1f} Y={model_rotation[1]:.1f} Z={model_rotation[2]:.1f}"
        cv2.putText(combined_frame, rot_text, (20, actual_height - 60), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        scale_text = f"Scale: {model_scale:.2f}"
        cv2.putText(combined_frame, scale_text, (20, actual_height - 30), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        # Show controls help
        controls_text = "Controls: Open hand=Move | Fist=Rotate | Two hands=Scale | ESC=Exit"
        cv2.putText(combined_frame, controls_text, 
                    (int(actual_width/2) - 300, actual_height - 30), 
                    UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)

        # Show the combined frame
        cv2.imshow(WINDOW_NAME, combined_frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("ESC pressed. Exiting...")
            break
        elif key == ord('r'):  # Reset transformations
            model_translation = [0.0, 0.0, -5.0]
            model_rotation = [0.0, 0.0, 0.0]
            model_scale = DEFAULT_MODEL_SCALE
            print("Transformations reset.")

    # --- Cleanup ---
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("AetherManipulator terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AetherManipulator - Manipulate 3D models with hand gestures")
    parser.add_argument("model_path", type=str, help="Path to OBJ model file")
    parser.add_argument("--scale", type=float, default=DEFAULT_MODEL_SCALE, help=f"Initial model scale (default: {DEFAULT_MODEL_SCALE})")
    
    args = parser.parse_args()
    
    # Set initial scale if provided
    DEFAULT_MODEL_SCALE = args.scale
    model_scale = DEFAULT_MODEL_SCALE
    
    main(args.model_path)
