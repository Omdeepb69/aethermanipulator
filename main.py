import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
import sys
import time
from ctypes import byref, c_int, CDLL, c_void_p
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pywavefront
import trimesh
try:
    import pymeshlab
except ImportError:
    pymeshlab = None

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "AetherManipulator"
DEFAULT_MODEL_SCALE = 1.0
MODEL_ROTATION_SPEED = 1.2
MODEL_TRANSLATION_SPEED = 2.0
MODEL_SCALE_SPEED = 0.08
UI_COLOR = (0, 255, 0)
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_SCALE = 0.7
UI_THICKNESS = 2

# Global state
interaction_mode = "TRANSLATE"
model_translation = [0.0, 0.0, -5.0]
model_rotation = [0.0, 0.0, 0.0]
model_scale = DEFAULT_MODEL_SCALE
model_type = None
model_meshes = None
model_materials = None
last_hand_centers = [None, None]
initial_pinch_distance = None
last_gesture_time = 0
gesture_lock = False
gl_context_created = False
gl_window_id = None

def initialize_glut():
    global gl_context_created, gl_window_id
    try:
        if not gl_context_created:
            # Try loading freeglut from local directory first
            if os.name == 'nt':
                freeglut_path = os.path.join(os.path.dirname(__file__), 'freeglut.dll')
                if os.path.exists(freeglut_path):
                    try:
                        CDLL(freeglut_path)
                    except Exception as e:
                        print(f"Warning: Could not load freeglut.dll: {e}")
            
            # Then try system-wide installation
            glutInit(sys.argv if hasattr(sys, 'argv') else [b''])
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
            glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
            glutInitWindowPosition(0, 0)
            gl_window_id = glutCreateWindow(b"AetherManipulator")
            gl_context_created = True
            init_opengl(WINDOW_WIDTH, WINDOW_HEIGHT)
        return True
    except Exception as e:
        print(f"GLUT initialization failed: {e}")
        print("Attempting alternative OpenGL context creation...")
        return create_alternative_opengl_context()

def create_alternative_opengl_context():
    global gl_context_created
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            
            # Windows-specific OpenGL context creation
            user32 = ctypes.WinDLL('user32', use_last_error=True)
            gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
            
            # Define necessary types and constants
            PIXELFORMATDESCRIPTOR = type('PIXELFORMATDESCRIPTOR', (ctypes.Structure,), {
                '_fields_': [
                    ('nSize', wintypes.WORD),
                    ('nVersion', wintypes.WORD),
                    ('dwFlags', wintypes.DWORD),
                    ('iPixelType', ctypes.c_byte),
                    ('cColorBits', ctypes.c_byte),
                    ('cRedBits', ctypes.c_byte),
                    ('cRedShift', ctypes.c_byte),
                    ('cGreenBits', ctypes.c_byte),
                    ('cGreenShift', ctypes.c_byte),
                    ('cBlueBits', ctypes.c_byte),
                    ('cBlueShift', ctypes.c_byte),
                    ('cAlphaBits', ctypes.c_byte),
                    ('cAlphaShift', ctypes.c_byte),
                    ('cAccumBits', ctypes.c_byte),
                    ('cAccumRedBits', ctypes.c_byte),
                    ('cAccumGreenBits', ctypes.c_byte),
                    ('cAccumBlueBits', ctypes.c_byte),
                    ('cAccumAlphaBits', ctypes.c_byte),
                    ('cDepthBits', ctypes.c_byte),
                    ('cStencilBits', ctypes.c_byte),
                    ('cAuxBuffers', ctypes.c_byte),
                    ('iLayerType', ctypes.c_byte),
                    ('bReserved', ctypes.c_byte),
                    ('dwLayerMask', wintypes.DWORD),
                    ('dwVisibleMask', wintypes.DWORD),
                    ('dwDamageMask', wintypes.DWORD),
                ]
            })
            
            # Create a dummy window
            hwnd = user32.CreateWindowExA(
                0, b"STATIC", b"Dummy OpenGL Window",
                0, 0, 0, 1, 1, None, None, None, None
            )
            
            if not hwnd:
                raise RuntimeError("Failed to create dummy window")
            
            hdc = user32.GetDC(hwnd)
            if not hdc:
                raise RuntimeError("Failed to get device context")
            
            # Set up pixel format
            pfd = PIXELFORMATDESCRIPTOR()
            pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
            pfd.nVersion = 1
            pfd.dwFlags = 0x25  # PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
            pfd.iPixelType = 0  # PFD_TYPE_RGBA
            pfd.cColorBits = 32
            pfd.cDepthBits = 24
            pfd.cStencilBits = 8
            pfd.iLayerType = 0  # PFD_MAIN_PLANE
            
            pixel_format = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
            if not pixel_format:
                raise RuntimeError("Failed to choose pixel format")
            
            if not gdi32.SetPixelFormat(hdc, pixel_format, ctypes.byref(pfd)):
                raise RuntimeError("Failed to set pixel format")
            
            # Create OpenGL context
            hglrc = user32.wglCreateContext(hdc)
            if not hglrc:
                raise RuntimeError("Failed to create OpenGL context")
            
            if not user32.wglMakeCurrent(hdc, hglrc):
                raise RuntimeError("Failed to make OpenGL context current")
            
            gl_context_created = True
            init_opengl(WINDOW_WIDTH, WINDOW_HEIGHT)
            return True
        return False
    except Exception as e:
        print(f"Alternative OpenGL context creation failed: {e}")
        return False

def init_opengl(width, height):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    light_ambient = [0.2, 0.2, 0.2, 1.0]
    light_diffuse = [0.8, 0.8, 0.8, 1.0]
    light_specular = [0.5, 0.5, 0.5, 1.0]
    light_position = [2.0, 2.0, 5.0, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    glShadeModel(GL_SMOOTH)
    glClearColor(0.1, 0.1, 0.15, 0.0)
    glClearDepth(1.0)

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def find_model_file(file_path):
    if os.path.isfile(file_path):
        return file_path
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alternative_path = os.path.join(script_dir, os.path.basename(file_path))
    
    if os.path.isfile(alternative_path):
        return alternative_path
    
    return None

def create_basic_cube(file_path):
    try:
        with open(file_path, 'w') as f:
            f.write("""# Simple cube OBJ file
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0

f 1 2 3 4
f 5 8 7 6
f 1 4 8 5
f 2 6 7 3
f 3 7 8 4
f 1 5 6 2
""")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create basic cube: {e}")
        return False

def load_model(model_file_path):
    global scene, model_scale, model_translation, model_type, model_meshes
    
    try:
        _, file_ext = os.path.splitext(model_file_path.lower())
        actual_path = find_model_file(model_file_path)
        
        if not actual_path:
            if os.path.basename(model_file_path).lower() == "cube.obj":
                script_dir = os.path.dirname(os.path.abspath(__file__))
                cube_path = os.path.join(script_dir, "cube.obj")
                if create_basic_cube(cube_path):
                    actual_path = cube_path
                    file_ext = ".obj"
                else:
                    raise FileNotFoundError(f"Could not create cube at {cube_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {model_file_path}")
        
        if file_ext == ".obj":
            model_type = "OBJ"
            scene = pywavefront.Wavefront(actual_path, collect_faces=True, parse=True)
            all_vertices = np.array([v for mesh in scene.mesh_list for v in mesh.vertices])
            
        elif file_ext == ".fbx":
            model_type = "FBX"
            if pymeshlab:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(actual_path)
                model_meshes = []
                for mesh_idx in range(ms.mesh_number()):
                    mesh = ms.mesh(mesh_idx)
                    model_meshes.append({
                        'vertices': mesh.vertex_matrix(),
                        'faces': mesh.face_matrix(),
                        'material': None
                    })
                all_vertices = np.vstack([mesh['vertices'] for mesh in model_meshes])
            else:
                mesh = trimesh.load(actual_path)
                if isinstance(mesh, trimesh.Scene):
                    model_meshes = []
                    for geometry_name, geometry in mesh.geometry.items():
                        transform = mesh.graph.get(geometry_name)[0]
                        vertices = np.array(geometry.vertices)
                        faces = np.array(geometry.faces)
                        if transform is not None:
                            vertices = trimesh.transformations.transform_points(vertices, transform)
                        model_meshes.append({
                            'vertices': vertices,
                            'faces': faces,
                            'material': None
                        })
                    all_vertices = np.vstack([mesh['vertices'] for mesh in model_meshes])
                else:
                    model_meshes = [{
                        'vertices': np.array(mesh.vertices),
                        'faces': np.array(mesh.faces),
                        'material': None
                    }]
                    all_vertices = np.array(mesh.vertices)
            scene = None
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        center = (min_coords + max_coords) / 2.0
        size = np.max(max_coords - min_coords)
        model_scale = DEFAULT_MODEL_SCALE / max(size, 1e-6)
        model_translation = [-center[0]*model_scale, -center[1]*model_scale, -5.0]
        
        print(f"Loaded {model_type} model: {os.path.basename(actual_path)}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_file_path}': {e}")
        return None

def draw_model():
    global model_type, model_meshes, scene
    
    if model_type is None:
        return
        
    glPushMatrix()
    glTranslatef(model_translation[0], model_translation[1], model_translation[2])
    glRotatef(model_rotation[0], 1.0, 0.0, 0.0)
    glRotatef(model_rotation[1], 0.0, 1.0, 0.0)
    glRotatef(model_rotation[2], 0.0, 0.0, 1.0)
    glScalef(model_scale, model_scale, model_scale)

    if model_type == "OBJ" and scene:
        for mesh in scene.mesh_list:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_index in face:
                    if vertex_index < len(scene.vertices):
                        vertex = scene.vertices[vertex_index]
                        if len(vertex) >= 3:
                            glNormal3f(0.0, 0.0, 1.0)
                            glVertex3fv(vertex[:3])
            glEnd()
    
    elif model_type == "FBX" and model_meshes:
        for mesh in model_meshes:
            vertices = mesh['vertices']
            faces = mesh['faces']
            
            glBegin(GL_TRIANGLES)
            for face in faces:
                if len(face) >= 3:
                    v0 = vertices[face[0]]
                    v1 = vertices[face[1]]
                    v2 = vertices[face[2]]
                    
                    normal = np.cross(v1-v0, v2-v0)
                    normal = normal/np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0,0,1]
                    
                    glNormal3fv(normal)
                    for vertex_idx in face:
                        if vertex_idx < len(vertices):
                            glVertex3fv(vertices[vertex_idx])
            glEnd()

    glPopMatrix()

def render_gl_scene(width, height):
    if not gl_context_created:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_model()

        glReadBuffer(GL_BACK)
        pixels = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image = cv2.flip(image, 0)
        
        glutSwapBuffers()
        return image
        
    except Exception as e:
        print(f"Error in render_gl_scene: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_landmarks_list(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    landmarks = []
    if hand_landmarks:
        for lm in hand_landmarks.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))
    return landmarks

def get_hand_center(landmarks_list):
    if not landmarks_list:
        return None
    x_coords = [lm[0] for lm in landmarks_list]
    y_coords = [lm[1] for lm in landmarks_list]
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    z_coords = [landmarks_list[i][2] for i in [0, 5, 9, 13, 17] if i < len(landmarks_list)]
    center_z = np.mean(z_coords) if z_coords else 0.0
    return (center_x, center_y, center_z)

def is_fist(landmarks_list):
    if not landmarks_list or len(landmarks_list) < 21:
        return False

    try:
        wrist = np.array(landmarks_list[0][:2])
        mcp_9 = np.array(landmarks_list[9][:2])
        palm_center = (wrist + mcp_9) / 2

        fingertip_distances = [np.linalg.norm(np.array(landmarks_list[tip][:2]) - palm_center) for tip in [8,12,16,20]]
        knuckle_distances = [np.linalg.norm(np.array(landmarks_list[knuckle][:2]) - palm_center) for knuckle in [5,9,13,17]]
        
        fist_ratio = sum(fingertip_distances) / (sum(knuckle_distances) + 1e-6)
        thumb_distance = np.linalg.norm(np.array(landmarks_list[4][:2]) - np.array(landmarks_list[5][:2]))
        hand_width = np.linalg.norm(np.array(landmarks_list[5][:2]) - np.array(landmarks_list[17][:2]))
        
        return fist_ratio < 0.85 and thumb_distance < 0.45 * hand_width
    except Exception:
        return False

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def calculate_2d_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_pinch_gesture(landmarks_list):
    if not landmarks_list or len(landmarks_list) < 21:
        return False
    
    try:
        thumb_tip = landmarks_list[4]
        index_tip = landmarks_list[8]
        distance = calculate_2d_distance(thumb_tip, index_tip)
        
        wrist = landmarks_list[0]
        middle_mcp = landmarks_list[9]
        hand_size = calculate_2d_distance(wrist, middle_mcp)
        
        return distance < 0.15 * hand_size
    except Exception:
        return False

def process_hands(frame, results, flip_coordinates=True):
    detected_hands = []
    
    if results.multi_hand_landmarks:
        annotated_frame = frame.copy()
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            landmarks_list = get_landmarks_list(hand_landmarks, frame.shape)
            if landmarks_list:
                handedness = results.multi_handedness[i].classification[0].label
                if flip_coordinates:
                    handedness = "Right" if handedness == "Left" else "Left"
                
                detected_hands.append({
                    "landmarks": landmarks_list,
                    "center": get_hand_center(landmarks_list),
                    "handedness": handedness,
                    "is_fist": is_fist(landmarks_list),
                    "is_pinch": is_pinch_gesture(landmarks_list)
                })
        
        return annotated_frame, detected_hands
    
    return frame, detected_hands

def main(model_path):
    global scene, interaction_mode, model_translation, model_rotation, model_scale
    global last_hand_centers, initial_pinch_distance, last_gesture_time, gesture_lock
    global gl_context_created

    print("Initializing AetherManipulator...")
    
    if not initialize_glut():
        print("Warning: OpenGL context creation failed - 3D rendering may not work")
        print("Please ensure you have freeglut installed on your system")
        print("On Windows, download freeglut.dll and place it in:")
        print("1. The same directory as this script")
        print("2. Your system32 folder (C:\\Windows\\System32)")
        print("3. Any directory in your system PATH")
    
    if not load_model(model_path):
        print("\nModel loading failed. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_width, actual_height)
    
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2)
    
    print("Starting real-time loop...")

    prev_time = time.time()
    smoothing_factor = 0.2
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / delta_time if delta_time > 0 else 0

        mirrored_frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        processed_frame, detected_hands = process_hands(mirrored_frame, results, flip_coordinates=False)

        num_hands = len(detected_hands)
        current_hand_centers = [None, None]
        
        if num_hands >= 1:
            detected_hands.sort(key=lambda x: x["handedness"])
            
            if num_hands == 1 and detected_hands[0]["is_pinch"] and not gesture_lock and (current_time - last_gesture_time) > 0.5:
                modes = ["TRANSLATE", "ROTATE", "SCALE"]
                current_index = modes.index(interaction_mode) if interaction_mode in modes else 0
                interaction_mode = modes[(current_index + 1) % len(modes)]
                print(f"Mode changed to: {interaction_mode}")
                last_gesture_time = current_time
                gesture_lock = True
            
            if num_hands == 1 and not detected_hands[0]["is_pinch"]:
                gesture_lock = False
            
            for i, hand in enumerate(detected_hands[:2]):
                current_hand_centers[i] = hand["center"]
                
                if interaction_mode == "TRANSLATE" and last_hand_centers[i] and current_hand_centers[i]:
                    delta_x = current_hand_centers[i][0] - last_hand_centers[i][0]
                    delta_y = current_hand_centers[i][1] - last_hand_centers[i][1]
                    
                    target_x = model_translation[0] + delta_x * MODEL_TRANSLATION_SPEED / actual_width
                    target_y = model_translation[1] - delta_y * MODEL_TRANSLATION_SPEED / actual_height
                    
                    model_translation[0] += (target_x - model_translation[0]) * smoothing_factor
                    model_translation[1] += (target_y - model_translation[1]) * smoothing_factor
                
                elif interaction_mode == "ROTATE" and last_hand_centers[i] and current_hand_centers[i] and not hand["is_pinch"]:
                    delta_x = current_hand_centers[i][0] - last_hand_centers[i][0]
                    delta_y = current_hand_centers[i][1] - last_hand_centers[i][1]
                    
                    target_yaw = model_rotation[1] + delta_x * MODEL_ROTATION_SPEED
                    target_pitch = model_rotation[0] + delta_y * MODEL_ROTATION_SPEED
                    
                    model_rotation[1] += (target_yaw - model_rotation[1]) * smoothing_factor
                    model_rotation[0] += (target_pitch - model_rotation[0]) * smoothing_factor
                
                elif interaction_mode == "SCALE" and last_hand_centers[i] and current_hand_centers[i]:
                    delta_y = current_hand_centers[i][1] - last_hand_centers[i][1]
                    target_scale = model_scale - delta_y * MODEL_SCALE_SPEED / actual_height
                    target_scale = max(0.1, min(5.0, target_scale))
                    model_scale += (target_scale - model_scale) * smoothing_factor
        
        if num_hands == 2 and all(hand["is_pinch"] for hand in detected_hands[:2]) and all(current_hand_centers):
            if all(last_hand_centers) and initial_pinch_distance is None:
                initial_pinch_distance = calculate_distance(current_hand_centers[0], current_hand_centers[1])
            
            if initial_pinch_distance:
                current_distance = calculate_distance(current_hand_centers[0], current_hand_centers[1])
                if current_distance > 0 and initial_pinch_distance > 0:
                    scale_factor = current_distance / initial_pinch_distance
                    target_scale = model_scale * scale_factor
                    target_scale = max(0.1, min(5.0, target_scale))
                    model_scale = target_scale
                    initial_pinch_distance = current_distance
        else:
            initial_pinch_distance = None
        
        last_hand_centers = current_hand_centers
        
        model_frame = render_gl_scene(actual_width, actual_height)
        combined_frame = cv2.addWeighted(processed_frame, 0.7, model_frame, 0.3, 0)
        
        cv2.putText(combined_frame, f"FPS: {int(fps)}", (20, 30), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        cv2.putText(combined_frame, f"Mode: {interaction_mode}", (20, 70), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        
        gesture_text = ""
        if num_hands == 0:
            gesture_text = "No hands detected"
        elif num_hands == 1:
            if detected_hands[0]["is_pinch"]:
                gesture_text = "Pinch detected (Change mode)"
            elif detected_hands[0]["is_fist"]:
                gesture_text = "Fist detected"
            else:
                gesture_text = f"Manipulating in {interaction_mode} mode"
        elif num_hands == 2:
            if all(hand["is_pinch"] for hand in detected_hands[:2]):
                gesture_text = "Two pinches detected (Scaling)"
            else:
                gesture_text = "Two hands detected"
        
        cv2.putText(combined_frame, gesture_text, (20, 110), UI_FONT, UI_SCALE, UI_COLOR, UI_THICKNESS)
        cv2.putText(combined_frame, "Q: Quit | R: Reset Rotation | T: Reset Position | S: Reset Scale", (20, 150), UI_FONT, 0.5, UI_COLOR, 1)
        
        cv2.imshow(WINDOW_NAME, combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            model_rotation = [0.0, 0.0, 0.0]
        elif key == ord('t'):
            model_translation = [0.0, 0.0, -5.0]
        elif key == ord('s'):
            model_scale = DEFAULT_MODEL_SCALE
        elif key == ord('m'):
            modes = ["TRANSLATE", "ROTATE", "SCALE"]
            current_index = modes.index(interaction_mode) if interaction_mode in modes else 0
            interaction_mode = modes[(current_index + 1) % len(modes)]
            print(f"Mode changed to: {interaction_mode}")
    
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    
    if gl_context_created:
        try:
            glutDestroyWindow(gl_window_id)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AetherManipulator: 3D model manipulation using hand gestures")
    parser.add_argument('--model', default='cube.obj', help='Path to 3D model file (.obj or .fbx)')
    args = parser.parse_args()
    
    try:
        main(args.model)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()