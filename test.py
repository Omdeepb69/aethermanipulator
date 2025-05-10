import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import time
import math
import mediapipe as mp

class ARHandTracker:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open webcam")
            sys.exit()
            
        # Get webcam dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize pygame
        pygame.init()
        self.display = (800, 600)
        pygame.display.set_caption("AR Hand Tracking - Press ESC to quit")
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        
        # Create a separate surface for the webcam feed
        self.background_surface = pygame.Surface((self.display[0], self.display[1]))
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)
        
        # Enable depth test and lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light position and properties
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize 3D objects
        self.init_objects()
        
        # Start time for animation
        self.start_time = time.time()
        
        # Hand position tracking
        self.hand_positions = []
        self.hand_gestures = []
        self.is_grabbing = False
        
        # For hybrid rendering (2D + 3D)
        self.clock = pygame.time.Clock()
        
    def init_objects(self):
        # Define cube vertices
        self.vertices = (
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (-1, -1, 1)
        )
        
        # Define cube surfaces (quads)
        self.surfaces = (
            (0, 1, 2, 3),  # back
            (3, 2, 6, 7),  # left
            (7, 6, 5, 4),  # front
            (4, 5, 1, 0),  # right
            (1, 5, 6, 2),  # top
            (4, 0, 3, 7)   # bottom
        )
        
        # Define colors for each surface with alpha (transparency)
        self.colors = (
            (0, 0, 1, 0.7),    # Blue
            (0, 1, 0, 0.7),    # Green
            (1, 0, 0, 0.7),    # Red
            (1, 1, 0, 0.7),    # Yellow
            (1, 0, 1, 0.7),    # Magenta
            (0, 1, 1, 0.7)     # Cyan
        )
        
        # Object position and rotation
        self.obj_position = [0, 0, 0]
        self.obj_rotation = [0, 0, 0]
        self.obj_scale = 0.5
    
    def draw_cube(self, position, rotation, scale=1.0):
        """Draw a colored cube at the specified position"""
        glPushMatrix()
        
        # Position and scale
        glTranslatef(position[0], position[1], position[2])
        
        # Apply rotation
        glRotatef(rotation[0], 1, 0, 0)  # X-axis
        glRotatef(rotation[1], 0, 1, 0)  # Y-axis
        glRotatef(rotation[2], 0, 0, 1)  # Z-axis
        
        # Apply scaling
        glScalef(scale, scale, scale)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw each face of the cube
        glBegin(GL_QUADS)
        for i, surface in enumerate(self.surfaces):
            glColor4fv(self.colors[i])
            for vertex in surface:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        
        # Disable blending
        glDisable(GL_BLEND)
        
        glPopMatrix()
    
    def process_hands(self, frame):
        """Process hand landmarks using MediaPipe"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        results = self.hands.process(frame_rgb)
        
        # Reset hand positions
        self.hand_positions = []
        self.hand_gestures = []
        
        # Check if hands were found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the 3D coordinates of the hand landmarks
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * self.width
                    y = landmark.y * self.height
                    z = landmark.z * self.width  # Scale z to match x scale
                    landmarks_3d.append((x, y, z))
                
                # Store hand position (using index finger tip)
                index_finger_pos = landmarks_3d[8]  # Index finger tip
                thumb_pos = landmarks_3d[4]  # Thumb tip
                
                # Convert to normalized coordinates (-1 to 1)
                norm_x = (index_finger_pos[0] / self.width) * 2 - 1
                norm_y = -((index_finger_pos[1] / self.height) * 2 - 1)  # Flip y-axis
                norm_z = index_finger_pos[2] / self.width
                
                # Store hand position
                self.hand_positions.append((norm_x * 3, norm_y * 3, norm_z))
                
                # Detect grabbing gesture (thumb and index finger close together)
                distance = np.sqrt(
                    (thumb_pos[0] - index_finger_pos[0])**2 + 
                    (thumb_pos[1] - index_finger_pos[1])**2 + 
                    (thumb_pos[2] - index_finger_pos[2])**2
                )
                
                # If distance is less than a threshold, consider it a grabbing gesture
                is_grabbing = distance < 50  # Adjust threshold as needed
                self.hand_gestures.append(is_grabbing)
                
                # Draw hand landmarks for debugging
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame_rgb
    
    def update_object_position(self):
        """Update object position based on hand tracking"""
        if len(self.hand_positions) > 0:
            # Get the first hand position
            hand_pos = self.hand_positions[0]
            
            # If grabbing, move the object
            if len(self.hand_gestures) > 0 and self.hand_gestures[0]:
                self.obj_position = [hand_pos[0], hand_pos[1], hand_pos[2]]
                self.is_grabbing = True
            elif self.is_grabbing:
                # Just released, add some velocity for a natural feel
                self.is_grabbing = False
    
    def run(self):
        """Main loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            try:
                # Capture webcam frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    continue
                    
                # Mirror the frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Process hands with MediaPipe
                frame_rgb = self.process_hands(frame)
                
                # Update object position based on hand tracking
                self.update_object_position()
                
                # Resize frame to match display dimensions
                resized_frame = cv2.resize(frame_rgb, (self.display[0], self.display[1]))
                
                # Convert to pygame surface
                pygame_surface = pygame.surfarray.make_surface(resized_frame.swapaxes(0, 1))
                
                # Display the pygame surface as background
                self.screen.blit(pygame_surface, (0, 0))
                
                # Calculate time-based rotation
                current_time = time.time() - self.start_time
                rotation_speed = 20
                
                if not self.is_grabbing:
                    # Only auto-rotate when not being grabbed
                    self.obj_rotation = [
                        (current_time * rotation_speed) % 360,
                        (current_time * rotation_speed * 0.7) % 360,
                        (current_time * rotation_speed * 0.5) % 360
                    ]
                
                # Enable 3D rendering
                glPushMatrix()
                
                # Clear depth buffer only (keep color buffer to keep webcam background)
                glClear(GL_DEPTH_BUFFER_BIT)
                
                # Draw 3D objects
                self.draw_cube(self.obj_position, self.obj_rotation, self.obj_scale)
                
                # Restore matrix
                glPopMatrix()
                
                # Update the display
                pygame.display.flip()
                
                # Control frame rate
                self.clock.tick(30)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                continue
        
        # Release resources
        self.cap.release()
        self.hands.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        app = ARHandTracker()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        pygame.quit()
        sys.exit(1)